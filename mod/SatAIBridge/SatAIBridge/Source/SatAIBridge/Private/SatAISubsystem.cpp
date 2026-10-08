#include "SatAISubsystem.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "EngineUtils.h"  
#include "Buildables/FGBuildable.h"
#include "Buildables/FGBuildableManufacturer.h"
#include "FGRecipe.h"
#include "FGFactoryConnectionComponent.h"
#include "Buildables/FGBuildableConveyorBelt.h"
#include "Buildables/FGBuildableStorage.h"
#include "FGInventoryComponent.h"
#include "Tests/FGTestBlueprintFunctionLibrary.h"
#include "HttpServerModule.h"
#include "IHttpRouter.h"
#include "HttpServerRequest.h"
#include "HttpServerResponse.h"
#include "Misc/ConfigCacheIni.h"
#include "IPAddress.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Pawn.h"
#include "FGLightweightBuildableSubsystem.h"
#include "AbstractInstanceInterface.h"
#include "InstanceData.h"
#include "Engine/StaticMesh.h"
#include "FGClearanceInterface.h"
#include "Resources/FGResourceNode.h"
#include "Resources/FGResourceDescriptor.h"
#include "Resources/FGItemDescriptor.h"
#include "FGWaterVolume.h"
#include "InstancedFoliageActor.h"
#include "WorldPartition/WorldPartitionSubsystem.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "FGDismantleInterface.h"

TWeakObjectPtr<ASatAISubsystem> ASatAISubsystem::Instance;

namespace
{
    void ForceLoopbackBinding(uint32 Port)
    {
        const TCHAR* Section = TEXT("HTTPServer.Listeners");
        const FString Override = FString::Printf(TEXT("(Port=%u,BindAddress=localhost)"), Port);

        TArray<FString> Overrides;
        GConfig->GetArray(Section, TEXT("ListenerOverrides"), Overrides, GEngineIni);
        if (!Overrides.Contains(Override))
        {
            Overrides.Insert(Override, 0);
            GConfig->SetArray(Section, TEXT("ListenerOverrides"), Overrides, GEngineIni);
        }
    }

    bool IsLocalRequest(const FHttpServerRequest& Request)
    {
        if (!Request.PeerAddress.IsValid())
        {
            return false;
        }
        const FString Ip = Request.PeerAddress->ToString(false);
        return Ip.StartsWith(TEXT("127.")) || Ip == TEXT("::1") || Ip.StartsWith(TEXT("::ffff:127."));
    }

    const TCHAR* DirectionName(EFactoryConnectionDirection Dir)
    {
        switch (Dir)
        {
        case EFactoryConnectionDirection::FCD_INPUT:  return TEXT("in");
        case EFactoryConnectionDirection::FCD_OUTPUT: return TEXT("out");
        default:                                      return TEXT("any");
        }
    }

    const TCHAR* ClearanceTypeName(EClearanceType Type)
    {
        switch (Type)
        {
        case EClearanceType::CT_Soft:           return TEXT("soft");
        case EClearanceType::CT_BlockEverything: return TEXT("block_everything");
        default:                                return TEXT("default");
        }
    }

    void CollectClearance(UObject* Source, TArray<FSatAIClearanceBox>& Out)
    {
        TArray<FFGClearanceData> Entries;
        IFGClearanceInterface::Execute_GetClearanceData(Source, Entries);
        for (const FFGClearanceData& Entry : Entries)
        {
            if (Entry.IsValid())
            {
                Out.Add({ ClearanceTypeName(Entry.Type), Entry.GetTransformedClearanceBox() });
            }
        }
    }

    constexpr double CmPerMetre = 100.0;

    TSharedPtr<FJsonObject> ParseJsonBody(const FHttpServerRequest& Request, FString& OutError)
    {
        if (Request.Body.Num() == 0)
        {
            OutError = TEXT("empty body");
            OutError = TEXT("empty body");
            return nullptr;
        }
        FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(Request.Body.GetData()), Request.Body.Num());
        const FString Text(Converted.Length(), Converted.Get());

        TSharedPtr<FJsonObject> Json;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
        if (!FJsonSerializer::Deserialize(Reader, Json) || !Json.IsValid())
        {
            OutError = TEXT("body is not a JSON object");
            return nullptr;
        }
        return Json;
    }

    bool ReadVec3Metres(const FJsonObject& Json, const FString& Field, FVector& OutCm, FString& OutError)
    {
        const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
        if (!Json.TryGetArrayField(Field, Arr) || Arr->Num() != 3)
        {
            OutError = FString::Printf(TEXT("%s must be [x, y, z]"), *Field);
            return false;
        }
        OutCm = FVector((*Arr)[0]->AsNumber(), (*Arr)[1]->AsNumber(), (*Arr)[2]->AsNumber()) * CmPerMetre;
        return true;
    }

    TArray<TSharedPtr<FJsonValue>> Vec3ToJsonMetres(const FVector& Cm)
    {
        const FVector M = Cm / CmPerMetre;
        return { MakeShared<FJsonValueNumber>(M.X), MakeShared<FJsonValueNumber>(M.Y), MakeShared<FJsonValueNumber>(M.Z) };
    }

    const TCHAR* NodeTypeName(EResourceNodeType NodeType)
    {
        switch (NodeType)
        {
        case EResourceNodeType::Node:              return TEXT("node");
        case EResourceNodeType::FrackingSatellite: return TEXT("fracking_satellite");
        case EResourceNodeType::FrackingCore:      return TEXT("fracking_core");
        case EResourceNodeType::Geyser:            return TEXT("geyser");
        case EResourceNodeType::Deposit:           return TEXT("deposit");
        default:                                   return TEXT("invalid");
        }
    }

    const TCHAR* PurityName(EResourcePurity Purity)
    {
        switch (Purity)
        {
        case RP_Inpure: return TEXT("impure");
        case RP_Normal: return TEXT("normal");
        case RP_Pure:   return TEXT("pure");
        default:        return TEXT("unknown");
        }
    }

    const TCHAR* FormName(EResourceForm Form)
    {
        switch (Form)
        {
        case EResourceForm::RF_SOLID:  return TEXT("solid");
        case EResourceForm::RF_LIQUID: return TEXT("liquid");
        case EResourceForm::RF_GAS:    return TEXT("gas");
        default:                       return TEXT("invalid");
        }
    }

    TSharedRef<FJsonObject> ResourceNodeToJson(const AFGResourceNodeBase& Node)
    {
        TSharedRef<FJsonObject> Json = MakeShared<FJsonObject>();
        Json->SetStringField(TEXT("id"), Node.GetName());

        const TSubclassOf<UFGResourceDescriptor> Resource = Node.GetResourceClass();
        Json->SetStringField(TEXT("resource"), Resource ? Resource->GetName() : TEXT("none"));
        Json->SetStringField(TEXT("type"), NodeTypeName(Node.GetResourceNodeType()));
        Json->SetStringField(TEXT("form"), FormName(Node.GetResourceForm()));
        Json->SetBoolField(TEXT("occupied"), Node.IsOccupied());

        if (const AFGResourceNode* WithPurity = Cast<AFGResourceNode>(&Node))
        {
            Json->SetStringField(TEXT("purity"), PurityName(WithPurity->GetResourcePurity()));
        }
        else
        {
            Json->SetField(TEXT("purity"), MakeShared<FJsonValueNull>());
        }

        Json->SetArrayField(TEXT("pos"), Vec3ToJsonMetres(Node.GetActorLocation()));
        return Json;
    }

    struct FSatAIWaterTester
    {
        TArray<TPair<FBox, AFGWaterVolume*>> Volumes;

        explicit FSatAIWaterTester(UWorld* World)
        {
            for (TActorIterator<AFGWaterVolume> It(World); It; ++It)
            {
                Volumes.Add({ It->GetComponentsBoundingBox(/*bNonColliding=*/ true), *It });
            }
        }

        bool IsUnderWater(const FVector& GroundCm) const
        {
            for (const TPair<FBox, AFGWaterVolume*>& Entry : Volumes)
            {
                const FBox& Box = Entry.Key;
                if (GroundCm.X < Box.Min.X || GroundCm.X > Box.Max.X ||
                    GroundCm.Y < Box.Min.Y || GroundCm.Y > Box.Max.Y)
                {
                    continue;
                }
                if (GroundCm.Z > Box.Max.Z)
                {
                    continue;
                }
                const FVector JustAboveGround = GroundCm + FVector(0, 0, 50);
                const FVector JustBelowSurface(GroundCm.X, GroundCm.Y, Box.Max.Z - 50);
                if (Entry.Value->EncompassesPoint(JustAboveGround) || Entry.Value->EncompassesPoint(JustBelowSurface))
                {
                    return true;
                }
            }
            return false;
        }
    };

    bool ParsePiece(const FJsonObject& Json, FSatAIPiece& Out, FString& OutError)
    {
        if (!Json.TryGetStringField(TEXT("id"), Out.Id) || !Json.TryGetStringField(TEXT("kind"), Out.Kind)
            || !Json.TryGetStringField(TEXT("class"), Out.ClassPath))
        {
            OutError = TEXT("needs id, kind and class");
            return false;
        }
        Json.TryGetStringField(TEXT("built_with"), Out.BuiltWith);

        if (Out.Kind == TEXT("belt"))
        {
            if (!Json.TryGetStringField(TEXT("from"), Out.From) || !Json.TryGetStringField(TEXT("to"), Out.To))
            {
                OutError = TEXT("belt needs from and to");
                return false;
            }
            Json.TryGetStringField(TEXT("from_port"), Out.FromPort);
            Json.TryGetStringField(TEXT("to_port"), Out.ToPort);
            return true;
        }

        const TSharedPtr<FJsonObject>* Fill = nullptr;
        if (Out.Kind == TEXT("container") && Json.TryGetObjectField(TEXT("fill"), Fill))
        {
            if (!(*Fill)->TryGetStringField(TEXT("item"), Out.FillItem)
                || !(*Fill)->TryGetNumberField(TEXT("amount"), Out.FillAmount) || Out.FillAmount <= 0)
            {
                OutError = TEXT("fill needs item and a positive amount");
                return false;
            }
        }

        FVector PosCm;
        if (!ReadVec3Metres(Json, TEXT("pos"), PosCm, OutError))
        {
            return false;
        }
        double Yaw = 0.0;
        Json.TryGetNumberField(TEXT("yaw"), Yaw);
        Out.Transform = FTransform(FRotator(0.0, Yaw, 0.0), PosCm);

        if (Out.Kind == TEXT("machine") && !Json.TryGetStringField(TEXT("recipe"), Out.Recipe))
        {
            OutError = TEXT("machine needs recipe");
            return false;
        }
        return true;
    }

    bool FillContainer(AFGBuildable* Buildable, const FString& ItemPath, int32 Amount, FString& OutError)
    {
        AFGBuildableStorage* Storage = Cast<AFGBuildableStorage>(Buildable);
        UFGInventoryComponent* Inventory = Storage ? Storage->GetStorageInventory() : nullptr;
        if (!Inventory)
        {
            OutError = TEXT("fill: not a storage container");
            return false;
        }
        TSubclassOf<UFGItemDescriptor> Item = LoadClass<UFGItemDescriptor>(nullptr, *ItemPath);
        if (!Item)
        {
            OutError = FString::Printf(TEXT("fill: item class not found: %s"), *ItemPath);
            return false;
        }
        const int32 StackSize = FMath::Max(1, UFGItemDescriptor::GetStackSize(Item));
        int32 Added = 0;
        while (Added < Amount)
        {
            const int32 Chunk = FMath::Min(StackSize, Amount - Added);
            const int32 Got = Inventory->AddStack(FInventoryStack(Chunk, Item), /*allowPartialAdd=*/ true);
            if (Got <= 0)
            {
                break;
            }
            Added += Got;
        }
        UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: filled %s with %d x %s"), *Storage->GetName(), Added, *Item->GetName());
        return true;
    }

    bool TraceGroundCm(UWorld* World, double XCm, double YCm, FVector& OutHitCm, FString& OutHitName)
    {
        FCollisionQueryParams Params(SCENE_QUERY_STAT(SatAIGround), /*bTraceComplex=*/ false);
        double RefZ = 0.0;
        if (APawn* Pawn = UGameplayStatics::GetPlayerPawn(World, 0))
        {
            Params.AddIgnoredActor(Pawn);
            RefZ = Pawn->GetActorLocation().Z;
        }

        FHitResult Hit;
        const FVector Start(XCm, YCm, RefZ + 10000.0);
        const FVector End(XCm, YCm, RefZ - 50000.0);
        if (!World->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params))
        {
            return false;
        }
        OutHitCm = Hit.ImpactPoint;
        OutHitName = Hit.GetActor() ? Hit.GetActor()->GetName() : TEXT("none");
        return true;
    }

    void RespondJson(const FHttpResultCallback& OnComplete, const TSharedRef<FJsonObject>& Json,
        EHttpServerResponseCodes Code = EHttpServerResponseCodes::Ok)
    {
        FString Body;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Body);
        FJsonSerializer::Serialize(Json, Writer);

        TUniquePtr<FHttpServerResponse> Response = FHttpServerResponse::Create(Body, TEXT("application/json"));
        Response->Code = Code;
        OnComplete(MoveTemp(Response));
    }

    void RespondError(const FHttpResultCallback& OnComplete, const FString& Message,
        EHttpServerResponseCodes Code)
    {
        TSharedRef<FJsonObject> Json = MakeShared<FJsonObject>();
        Json->SetBoolField(TEXT("ok"), false);
        Json->SetStringField(TEXT("error"), Message);
        RespondJson(OnComplete, Json, Code);
    }

    using FJsonHandler = TFunction<bool(ASatAISubsystem& Self, const FHttpServerRequest& Request,
        TSharedRef<FJsonObject>& OutJson, FString& OutError)>;

    FHttpRouteHandle BindJsonRoute(IHttpRouter& Router, const FString& Path, EHttpServerRequestVerbs Verb,
        TWeakObjectPtr<ASatAISubsystem> WeakSelf, FJsonHandler Handler)
    {
        FHttpRouteHandle Handle = Router.BindRoute(FHttpPath(Path), Verb,
            FHttpRequestHandler::CreateLambda(
                [WeakSelf, Handler, Path](const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
                {
                    const FString Peer = Request.PeerAddress.IsValid()
                        ? Request.PeerAddress->ToString(true) : TEXT("unknown");

                    if (!IsLocalRequest(Request))
                    {
                        UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: rejected HTTP %s from %s"), *Path, *Peer);
                        RespondError(OnComplete, TEXT("forbidden"), EHttpServerResponseCodes::Forbidden);
                        return true;
                    }

                    ASatAISubsystem* Self = WeakSelf.Get();
                    if (!Self)
                    {
                        RespondError(OnComplete, TEXT("subsystem gone"), EHttpServerResponseCodes::ServiceUnavail);
                        return true;
                    }

                    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: HTTP %s from %s"), *Path, *Peer);

                    TSharedRef<FJsonObject> Json = MakeShared<FJsonObject>();
                    FString Error;
                    if (!Handler(*Self, Request, Json, Error))
                    {
                        RespondError(OnComplete, Error, EHttpServerResponseCodes::BadRequest);
                        return true;
                    }

                    Json->SetBoolField(TEXT("ok"), true);
                    RespondJson(OnComplete, Json);
                    return true;
                }));

        if (!Handle.IsValid())
        {
            UE_LOG(LogTemp, Error, TEXT("SatAIBridge: failed to bind %s (already bound?)"), *Path);
        }
        return Handle;
    }
}

void ASatAISubsystem::BeginPlay()
{
    Super::BeginPlay();
    Instance = this;
    StartHttpServer();
    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: subsystem started"));
}

void ASatAISubsystem::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    StopHttpServer();
    if (Instance.Get() == this)
    {
        Instance.Reset();
    }
    Super::EndPlay(EndPlayReason);
}

void ASatAISubsystem::StartHttpServer()
{
    ForceLoopbackBinding(HttpPort);

    FHttpServerModule& Http = FHttpServerModule::Get();
    HttpRouter = Http.GetHttpRouter(HttpPort, /*bFailOnBindFailure=*/ true);
    if (!HttpRouter.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("SatAIBridge: could not get HTTP router on port %u"), HttpPort);
        return;
    }

    TWeakObjectPtr<ASatAISubsystem> WeakThis(this);
    auto AddRoute = [this](const FHttpRouteHandle& Handle)
        {
            if (Handle.IsValid())
            {
                HttpRoutes.Add(Handle);
            }
        };

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/ping"), EHttpServerRequestVerbs::VERB_GET, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            Out->SetStringField(TEXT("message"), Self.HandlePing());
            Out->SetNumberField(TEXT("ping_count"), Self.PingCount);
            Out->SetStringField(TEXT("version"), TEXT("0.1"));
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/player"), EHttpServerRequestVerbs::VERB_GET, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            APawn* Pawn = UGameplayStatics::GetPlayerPawn(Self.GetWorld(), 0);
            if (!Pawn)
            {
                OutError = TEXT("no player pawn");
                return false;
            }
            Out->SetArrayField(TEXT("pos"), Vec3ToJsonMetres(Pawn->GetActorLocation()));
            Out->SetNumberField(TEXT("yaw"), Pawn->GetControlRotation().Yaw);  // camera direction
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/ground"), EHttpServerRequestVerbs::VERB_GET, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            const FString* X = Request.QueryParams.Find(TEXT("x"));
            const FString* Y = Request.QueryParams.Find(TEXT("y"));
            if (!X || !Y)
            {
                OutError = TEXT("need ?x=..&y=.. in metres");
                return false;
            }
            FVector HitCm;
            FString HitName;
            if (!TraceGroundCm(Self.GetWorld(), FCString::Atod(**X) * CmPerMetre, FCString::Atod(**Y) * CmPerMetre,
                HitCm, HitName))
            {
                OutError = TEXT("no ground found");
                return false;
            }
            Out->SetNumberField(TEXT("z"), HitCm.Z / CmPerMetre);
            Out->SetStringField(TEXT("hit"), HitName);
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/spawn"), EHttpServerRequestVerbs::VERB_POST, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            TSharedPtr<FJsonObject> Body = ParseJsonBody(Request, OutError);
            if (!Body)
            {
                return false;
            }
            FString ClassPath, BuildId, BuiltWith;
            if (!Body->TryGetStringField(TEXT("class"), ClassPath))
            {
                OutError = TEXT("missing class");
                return false;
            }
            if (!Body->TryGetStringField(TEXT("build_id"), BuildId))
            {
                OutError = TEXT("missing build_id");
                return false;
            }
            Body->TryGetStringField(TEXT("built_with"), BuiltWith);
            FVector PosCm;
            if (!ReadVec3Metres(*Body, TEXT("pos"), PosCm, OutError))
            {
                return false;
            }
            double Yaw = 0.0;
            Body->TryGetNumberField(TEXT("yaw"), Yaw);

            FSatAISpawnResult Result;
            if (!Self.SpawnTracked(BuildId, ClassPath, BuiltWith,
                FTransform(FRotator(0.0, Yaw, 0.0), PosCm), Result, OutError))
            {
                return false;
            }

            Out->SetStringField(TEXT("name"), Result.Name);
            Out->SetBoolField(TEXT("lightweight"), Result.bLightweight);
            Out->SetArrayField(TEXT("pos"), Vec3ToJsonMetres(Result.Transform.GetLocation()));
            Out->SetNumberField(TEXT("yaw"), Result.Transform.Rotator().Yaw);
            if (Result.LocalBounds.IsValid)
            {
                Out->SetArrayField(TEXT("bounds_min"), Vec3ToJsonMetres(Result.LocalBounds.Min));
                Out->SetArrayField(TEXT("bounds_max"), Vec3ToJsonMetres(Result.LocalBounds.Max));
            }
            else
            {
                Out->SetField(TEXT("bounds_min"), MakeShared<FJsonValueNull>());
                Out->SetField(TEXT("bounds_max"), MakeShared<FJsonValueNull>());
            }
            TArray<TSharedPtr<FJsonValue>> PortArray;
            for (const FSatAIPortGeometry& Geo : Result.Ports)
            {
                TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
                J->SetStringField(TEXT("name"), Geo.Name);
                J->SetStringField(TEXT("dir"), Geo.Direction);
                J->SetArrayField(TEXT("pos"), Vec3ToJsonMetres(Geo.LocalPosCm));
                J->SetArrayField(TEXT("facing"), TArray<TSharedPtr<FJsonValue>>{
                    MakeShared<FJsonValueNumber>(Geo.LocalFacing.X),
                        MakeShared<FJsonValueNumber>(Geo.LocalFacing.Y),
                        MakeShared<FJsonValueNumber>(Geo.LocalFacing.Z) });
                PortArray.Add(MakeShared<FJsonValueObject>(J));
            }
            Out->SetArrayField(TEXT("ports"), PortArray);
            TArray<TSharedPtr<FJsonValue>> ClearanceArray;
            for (const FSatAIClearanceBox& Box : Result.Clearance)
            {
                TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
                J->SetStringField(TEXT("type"), Box.Type);
                J->SetArrayField(TEXT("min"), Vec3ToJsonMetres(Box.LocalBox.Min));
                J->SetArrayField(TEXT("max"), Vec3ToJsonMetres(Box.LocalBox.Max));
                ClearanceArray.Add(MakeShared<FJsonValueObject>(J));
            }
            Out->SetArrayField(TEXT("clearance"), ClearanceArray);
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/clear"), EHttpServerRequestVerbs::VERB_POST, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            TSharedPtr<FJsonObject> Body = ParseJsonBody(Request, OutError);
            if (!Body)
            {
                return false;
            }
            FString BuildId;
            if (!Body->TryGetStringField(TEXT("build_id"), BuildId))
            {
                OutError = TEXT("missing build_id");
                return false;
            }
            Out->SetNumberField(TEXT("destroyed"), Self.ClearBuild(BuildId));
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/build"), EHttpServerRequestVerbs::VERB_POST, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            TSharedPtr<FJsonObject> Body = ParseJsonBody(Request, OutError);
            if (!Body)
            {
                return false;
            }
            FString BuildId;
            if (!Body->TryGetStringField(TEXT("build_id"), BuildId))
            {
                OutError = TEXT("missing build_id");
                return false;
            }
            const TArray<TSharedPtr<FJsonValue>>* PieceValues = nullptr;
            if (!Body->TryGetArrayField(TEXT("pieces"), PieceValues))
            {
                OutError = TEXT("missing pieces");
                return false;
            }

            TArray<FSatAIPiece> Pieces;
            for (int32 i = 0; i < PieceValues->Num(); ++i)
            {
                const TSharedPtr<FJsonObject>* Obj = nullptr;
                if (!(*PieceValues)[i]->TryGetObject(Obj))
                {
                    OutError = FString::Printf(TEXT("pieces[%d] is not an object"), i);
                    return false;
                }
                FString PieceError;
                if (!ParsePiece(**Obj, Pieces.AddDefaulted_GetRef(), PieceError))
                {
                    OutError = FString::Printf(TEXT("pieces[%d]: %s"), i, *PieceError);
                    return false;
                }
            }

            int32 Built = 0;
            if (!Self.BuildPieces(BuildId, Pieces, Built, OutError))
            {
                return false;
            }
            Out->SetStringField(TEXT("build_id"), BuildId);
            Out->SetNumberField(TEXT("built"), Built);
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/verify"), EHttpServerRequestVerbs::VERB_GET, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            const FString* BuildId = Request.QueryParams.Find(TEXT("build_id"));
            if (!BuildId)
            {
                OutError = TEXT("need ?build_id=...");
                return false;
            }
            TArray<FSatAIPieceReport> Reports;
            if (!Self.VerifyBuild(*BuildId, Reports, OutError))
            {
                return false;
            }

            TArray<TSharedPtr<FJsonValue>> PieceArray;
            for (const FSatAIPieceReport& R : Reports)
            {
                TSharedRef<FJsonObject> P = MakeShared<FJsonObject>();
                P->SetStringField(TEXT("id"), R.Id);
                P->SetStringField(TEXT("class"), R.ClassName);
                P->SetBoolField(TEXT("exists"), R.bExists);
                P->SetBoolField(TEXT("lightweight"), R.bLightweight);
                if (R.bExists)
                {
                    P->SetArrayField(TEXT("pos"), Vec3ToJsonMetres(R.Transform.GetLocation()));
                    P->SetNumberField(TEXT("yaw"), R.Transform.Rotator().Yaw);
                }

                TArray<TSharedPtr<FJsonValue>> PortArray;
                for (const FSatAIPortReport& Port : R.Ports)
                {
                    TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
                    J->SetStringField(TEXT("name"), Port.Name);
                    J->SetStringField(TEXT("dir"), Port.Direction);
                    J->SetBoolField(TEXT("connected"), Port.bConnected);
                    J->SetStringField(TEXT("to"), Port.ConnectedTo);
                    PortArray.Add(MakeShared<FJsonValueObject>(J));
                }
                P->SetArrayField(TEXT("ports"), PortArray);
                PieceArray.Add(MakeShared<FJsonValueObject>(P));
            }
            Out->SetStringField(TEXT("build_id"), *BuildId);
            Out->SetArrayField(TEXT("pieces"), PieceArray);
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/resource_nodes"), EHttpServerRequestVerbs::VERB_GET, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            TArray<TSharedPtr<FJsonValue>> NodeList;
            for (TActorIterator<AFGResourceNodeBase> It(Self.GetWorld()); It; ++It)
            {
                NodeList.Add(MakeShared<FJsonValueObject>(ResourceNodeToJson(**It)));
            }
            Out->SetArrayField(TEXT("nodes"), NodeList);
            Out->SetNumberField(TEXT("count"), NodeList.Num());
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/teleport"), EHttpServerRequestVerbs::VERB_POST, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            const TSharedPtr<FJsonObject> Body = ParseJsonBody(Request, OutError);
            FVector DestCm;
            if (!Body || !ReadVec3Metres(*Body, TEXT("pos"), DestCm, OutError))
            {
                return false;
            }
            bool bFreeze = false;
            Body->TryGetBoolField(TEXT("freeze"), bFreeze);

            APawn* Pawn = UGameplayStatics::GetPlayerPawn(Self.GetWorld(), 0);
            if (!Pawn)
            {
                OutError = TEXT("no player pawn");
                return false;
            }

            bool bFrozen = false;
            if (ACharacter* Character = Cast<ACharacter>(Pawn))
            {
                UCharacterMovementComponent* Movement = Character->GetCharacterMovement();
                Movement->StopMovementImmediately();
                if (bFreeze)
                {
                    Movement->DisableMovement();
                }
                else if (Movement->MovementMode == MOVE_None)
                {
                    Movement->SetMovementMode(MOVE_Falling);
                }
                bFrozen = Movement->MovementMode == MOVE_None;
            }

            if (!Pawn->TeleportTo(DestCm, Pawn->GetActorRotation()))
            {
                OutError = TEXT("teleport blocked");
                return false;
            }
            Out->SetArrayField(TEXT("pos"), Vec3ToJsonMetres(Pawn->GetActorLocation()));
            Out->SetBoolField(TEXT("frozen"), bFrozen);
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/sample_terrain"), EHttpServerRequestVerbs::VERB_POST, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            const TSharedPtr<FJsonObject> Body = ParseJsonBody(Request, OutError);
            if (!Body)
            {
                return false;
            }
            const TArray<TSharedPtr<FJsonValue>>* OriginArr = nullptr;
            int32 Nx = 0;
            int32 Ny = 0;
            if (!Body->TryGetArrayField(TEXT("origin"), OriginArr) || OriginArr->Num() != 2
                || !Body->TryGetNumberField(TEXT("nx"), Nx) || !Body->TryGetNumberField(TEXT("ny"), Ny))
            {
                OutError = TEXT("needs origin [x, y], nx, ny (optional: step, z_top, z_bottom; metres)");
                return false;
            }
            double StepM = 8.0;
            double TopM = 1500.0;
            double BottomM = -500.0;
            Body->TryGetNumberField(TEXT("step"), StepM);
            Body->TryGetNumberField(TEXT("z_top"), TopM);
            Body->TryGetNumberField(TEXT("z_bottom"), BottomM);

            constexpr int32 MaxSamples = 65536;
            if (Nx < 1 || Ny < 1 || static_cast<int64>(Nx) * Ny > MaxSamples || StepM == 0.0 || TopM <= BottomM)
            {
                OutError = FString::Printf(TEXT("need nx, ny >= 1, nx*ny <= %d, step != 0, z_top > z_bottom"), MaxSamples);
                return false;
            }

            UWorld* World = Self.GetWorld();
            FCollisionQueryParams Params(SCENE_QUERY_STAT(SatAITerrain), /*bTraceComplex=*/ false);
            if (APawn* Pawn = UGameplayStatics::GetPlayerPawn(World, 0))
            {
                Params.AddIgnoredActor(Pawn);
            }
            const FSatAIWaterTester Water(World);

            const double OriginXCm = (*OriginArr)[0]->AsNumber() * CmPerMetre;
            const double OriginYCm = (*OriginArr)[1]->AsNumber() * CmPerMetre;
            const double StepCm = StepM * CmPerMetre;

            TArray<TSharedPtr<FJsonValue>> Heights, HitIds, WaterFlags;
            TArray<FString> ClassNames;
            TMap<FString, int32> ClassIndex;
            int32 Missed = 0;

            for (int32 Iy = 0; Iy < Ny; ++Iy)
            {
                for (int32 Ix = 0; Ix < Nx; ++Ix)
                {
                    const double XCm = OriginXCm + Ix * StepCm;
                    const double YCm = OriginYCm + Iy * StepCm;
                    const FVector Start(XCm, YCm, TopM * CmPerMetre);
                    const FVector End(XCm, YCm, BottomM * CmPerMetre);
                    FHitResult Hit;
                    bool bHit = false;
                    for (int32 Attempt = 0; Attempt < 4; ++Attempt)
                    {
                        bHit = World->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params);
                        AInstancedFoliageActor* Foliage = bHit ? Cast<AInstancedFoliageActor>(Hit.GetActor()) : nullptr;
                        if (!Foliage)
                        {
                            break;
                        }
                        Params.AddIgnoredActor(Foliage);
                        bHit = false;
                    }
                    if (!bHit)
                    {
                        Heights.Add(MakeShared<FJsonValueNull>());
                        HitIds.Add(MakeShared<FJsonValueNumber>(-1));
                        WaterFlags.Add(MakeShared<FJsonValueBoolean>(false));
                        ++Missed;
                        continue;
                    }

                    const AActor* HitActor = Hit.GetActor();
                    const FString ClassName = HitActor ? HitActor->GetClass()->GetName() : TEXT("none");
                    const int32* Known = ClassIndex.Find(ClassName);
                    const int32 HitId = Known ? *Known : ClassIndex.Add(ClassName, ClassNames.Add(ClassName));

                    Heights.Add(MakeShared<FJsonValueNumber>(Hit.ImpactPoint.Z / CmPerMetre));
                    HitIds.Add(MakeShared<FJsonValueNumber>(HitId));
                    WaterFlags.Add(MakeShared<FJsonValueBoolean>(Water.IsUnderWater(Hit.ImpactPoint)));
                }
            }

            TArray<TSharedPtr<FJsonValue>> ClassList;
            for (const FString& Name : ClassNames)
            {
                ClassList.Add(MakeShared<FJsonValueString>(Name));
            }
            Out->SetArrayField(TEXT("z"), Heights);
            Out->SetArrayField(TEXT("hit"), HitIds);
            Out->SetArrayField(TEXT("water"), WaterFlags);
            Out->SetArrayField(TEXT("classes"), ClassList);
            Out->SetNumberField(TEXT("missed"), Missed);
            return true;
        }));

    AddRoute(BindJsonRoute(*HttpRouter, TEXT("/streaming_status"), EHttpServerRequestVerbs::VERB_GET, WeakThis,
        [](ASatAISubsystem& Self, const FHttpServerRequest& Request, TSharedRef<FJsonObject>& Out, FString& OutError)
        {
            UWorld* World = Self.GetWorld();
            const UWorldPartitionSubsystem* Partition = World->GetSubsystem<UWorldPartitionSubsystem>();
            Out->SetBoolField(TEXT("world_partition"), World->GetWorldPartition() != nullptr);
            Out->SetBoolField(TEXT("complete"), Partition ? Partition->IsStreamingCompleted() : true);
            return true;
        }));

    Http.StartAllListeners();
    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: HTTP server listening on 127.0.0.1:%u (%d routes)"),
        HttpPort, HttpRoutes.Num());
}

void ASatAISubsystem::StopHttpServer()
{
    if (HttpRouter.IsValid())
    {
        for (const FHttpRouteHandle& Route : HttpRoutes)
        {
            HttpRouter->UnbindRoute(Route);
        }
    }
    HttpRoutes.Empty();
    HttpRouter.Reset();
}

ASatAISubsystem* ASatAISubsystem::Get(const UObject* WorldContext)
{
    UWorld* World = GEngine->GetWorldFromContextObject(WorldContext, EGetWorldErrorMode::ReturnNull);
    if (!World) return nullptr;

    if (Instance.IsValid() && Instance->GetWorld() == World)
    {
        return Instance.Get();
    }

    for (TActorIterator<ASatAISubsystem> It(World); It; ++It)
    {
        Instance = *It;
        return *It;
    }
    return nullptr;
}

AFGBuildable* ASatAISubsystem::SpawnBuildable(const FString& ClassPath, const FTransform& Transform, FString& OutError,
    const FString& BuiltWithRecipePath)
{
    UClass* BuildClass = LoadClass<AFGBuildable>(nullptr, *ClassPath);
    if (!BuildClass)
    {
        OutError = FString::Printf(TEXT("class not found: %s"), *ClassPath);
        return nullptr;
    }

    TSubclassOf<UFGRecipe> BuiltWith;
    if (!BuiltWithRecipePath.IsEmpty())
    {
        BuiltWith = LoadClass<UFGRecipe>(nullptr, *BuiltWithRecipePath);
        if (!BuiltWith)
        {
            OutError = FString::Printf(TEXT("built_with recipe not found: %s"), *BuiltWithRecipePath);
            return nullptr;
        }
    }

    AFGBuildable* Buildable = GetWorld()->SpawnActorDeferred<AFGBuildable>(
        BuildClass, Transform, nullptr, nullptr,
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn);
    if (!Buildable)
    {
        OutError = TEXT("SpawnActorDeferred returned null");
        return nullptr;
    }

    if (BuiltWith)
    {
        Buildable->SetBuiltWithRecipe(BuiltWith);
    }

    Buildable->FinishSpawning(Transform);

    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: spawned %s at %s"),
        *Buildable->GetName(), *Transform.GetLocation().ToString());
    return Buildable;
}

bool ASatAISubsystem::SetMachineRecipe(AFGBuildable* Buildable, const FString& RecipePath, FString& OutError)
{
    AFGBuildableManufacturer* Machine = Cast<AFGBuildableManufacturer>(Buildable);
    if (!Machine)
    {
        OutError = TEXT("buildable is not a manufacturer");
        return false;
    }

    TSubclassOf<UFGRecipe> Recipe = LoadClass<UFGRecipe>(nullptr, *RecipePath);
    if (!Recipe)
    {
        OutError = FString::Printf(TEXT("recipe not found: %s"), *RecipePath);
        return false;
    }

    Machine->SetRecipe(Recipe);
    return true;
}

void ASatAISubsystem::LogPorts(AFGBuildable* Buildable) const
{
    TArray<UFGFactoryConnectionComponent*> Ports;
    Buildable->GetComponents<UFGFactoryConnectionComponent>(Ports);

    const FTransform& BuildingTransform = Buildable->GetActorTransform();
    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: %s has %d belt ports"),
        *Buildable->GetClass()->GetName(), Ports.Num());

    for (UFGFactoryConnectionComponent* Port : Ports)
    {
        const EFactoryConnectionDirection Dir = Port->GetDirection();
        const TCHAR* DirText =
            Dir == EFactoryConnectionDirection::FCD_INPUT ? TEXT("IN") :
            Dir == EFactoryConnectionDirection::FCD_OUTPUT ? TEXT("OUT") : TEXT("OTHER");

        FVector LocalPos = BuildingTransform.InverseTransformPosition(Port->GetComponentLocation()) / 100.f;
        FVector LocalFacing = BuildingTransform.InverseTransformVectorNoScale(Port->GetConnectorNormal());

        UE_LOG(LogTemp, Warning, TEXT("SatAIBridge:   %-20s %-5s pos_m=%s facing=%s"),
            *Port->GetName(), DirText, *LocalPos.ToString(), *LocalFacing.ToString());
    }
}

UFGFactoryConnectionComponent* ASatAISubsystem::FindFreePort(AFGBuildable* Buildable, bool bOutput,
    const FString& PortName) const
{
    const EFactoryConnectionDirection Wanted =
        bOutput ? EFactoryConnectionDirection::FCD_OUTPUT : EFactoryConnectionDirection::FCD_INPUT;

    TArray<UFGFactoryConnectionComponent*> Ports;
    Buildable->GetComponents<UFGFactoryConnectionComponent>(Ports);
    for (UFGFactoryConnectionComponent* Port : Ports)
    {
        if (Port->GetDirection() == Wanted && !Port->IsConnected()
            && (PortName.IsEmpty() || Port->GetName() == PortName))
        {
            return Port;
        }
    }
    return nullptr;
}

AFGBuildableConveyorBelt* ASatAISubsystem::ConnectWithBelt(AFGBuildable* From, AFGBuildable* To,
    const FString& BeltClassPath, FString& OutError, const FString& FromPort, const FString& ToPort)
{
    UFGFactoryConnectionComponent* OutPort = FindFreePort(From, true, FromPort);
    UFGFactoryConnectionComponent* InPort = FindFreePort(To, false, ToPort);
    if (!OutPort || !InPort)
    {
        OutError = FString::Printf(TEXT("no free output '%s' on source or no free input '%s' on target"),
            *FromPort, *ToPort);
        return nullptr;
    }

    TSubclassOf<AFGBuildable> BeltClass = LoadClass<AFGBuildableConveyorBelt>(nullptr, *BeltClassPath);
    if (!BeltClass)
    {
        OutError = FString::Printf(TEXT("belt class not found: %s"), *BeltClassPath);
        return nullptr;
    }

    AFGBuildableConveyorBelt* Belt = Cast<AFGBuildableConveyorBelt>(
        UFGTestBlueprintFunctionLibrary::SpawnSplineBuildable(BeltClass, OutPort, InPort));
    if (!Belt)
    {
        OutError = TEXT("SpawnSplineBuildable returned null");
        return nullptr;
    }

    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: belt %s from %s.%s to %s.%s, length %.1f m"),
        *Belt->GetName(),
        *From->GetName(), *OutPort->GetName(),
        *To->GetName(), *InPort->GetName(),
        Belt->GetLength() / 100.f);
    return Belt;
}

FString ASatAISubsystem::HandlePing()
{
    PingCount++;
    return FString::Printf(TEXT("SatAIBridge: pong (v0.1, ping #%d)"), PingCount);
}

bool ASatAISubsystem::SpawnTracked(const FString& BuildId, const FString& ClassPath, const FString& BuiltWithRecipePath,
    const FTransform& Transform, FSatAISpawnResult& Out, FString& OutError)
{
    AFGBuildable* Built = SpawnBuildable(ClassPath, Transform, OutError, BuiltWithRecipePath);
    if (!Built)
    {
        return false;
    }
    Out.Name = Built->GetName();
    UClass* Class = Built->GetClass();

    if (IsValid(Built))
    {
        Out.Actor = Built;
        Out.Transform = Built->GetActorTransform();
        Out.LocalBounds = Built->CalculateComponentsBoundingBoxInLocalSpace(/*bNonColliding=*/ false);
        if (!Out.LocalBounds.IsValid)
        {
            const TArray<FInstanceData> MeshInstances =
                IAbstractInstanceInterface::Execute_GetActorLightweightInstanceData(Built);
            for (const FInstanceData& MeshInstance : MeshInstances)
            {
                if (MeshInstance.StaticMesh)
                {
                    Out.LocalBounds += MeshInstance.StaticMesh->GetBoundingBox().TransformBy(MeshInstance.RelativeTransform);
                }
            }
        }
        CollectClearance(Built, Out.Clearance);
        TArray<UFGFactoryConnectionComponent*> Ports;
        Built->GetComponents<UFGFactoryConnectionComponent>(Ports);
        for (UFGFactoryConnectionComponent* Port : Ports)
        {
            FSatAIPortGeometry& Geo = Out.Ports.AddDefaulted_GetRef();
            Geo.Name = Port->GetName();
            Geo.Direction = DirectionName(Port->GetDirection());
            Geo.LocalPosCm = Out.Transform.InverseTransformPosition(Port->GetComponentLocation());
            Geo.LocalFacing = Out.Transform.InverseTransformVectorNoScale(Port->GetConnectorNormal());
        }
        Builds.FindOrAdd(BuildId).Add({ Built, nullptr, INDEX_NONE, Out.Transform.GetLocation() });
        return true;
    }

    AFGLightweightBuildableSubsystem* Lightweight = AFGLightweightBuildableSubsystem::Get(GetWorld());
    const int32 Index = FindLightweightIndex(Class, Transform.GetLocation());
    const FRuntimeBuildableInstanceData* Data = (Lightweight && Index != INDEX_NONE)
        ? Lightweight->GetRuntimeDataForBuildableClassAndIndex(Class, Index) : nullptr;
    if (!Data)
    {
        OutError = TEXT("spawned, but it became a lightweight instance that could not be found");
        return false;
    }

    Out.bLightweight = true;
    Out.Transform = Data->Transform;
    Out.LocalBounds = Data->BoundingBox;
    CollectClearance(Class->GetDefaultObject(), Out.Clearance);
    Builds.FindOrAdd(BuildId).Add({ nullptr, Class, Index, Data->Transform.GetLocation() });
    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: %s became lightweight instance #%d"), *Out.Name, Index);
    return true;
}

int32 ASatAISubsystem::FindLightweightIndex(UClass* Class, const FVector& LocationCm) const
{
    AFGLightweightBuildableSubsystem* Lightweight = AFGLightweightBuildableSubsystem::Get(GetWorld());
    if (!Lightweight)
    {
        return INDEX_NONE;
    }
    const TArray<FRuntimeBuildableInstanceData>* Instances =
        Lightweight->GetAllLightweightBuildableInstances().Find(Class);
    if (!Instances)
    {
        return INDEX_NONE;
    }
    for (int32 i = Instances->Num() - 1; i >= 0; --i)
    {
        const FRuntimeBuildableInstanceData& Data = (*Instances)[i];
        if (Data.Handles.Num() > 0 && Data.Transform.GetLocation().Equals(LocationCm, 1.0))
        {
            return i;
        }
    }
    return INDEX_NONE;
}

int32 ASatAISubsystem::ClearBuild(const FString& BuildId)
{
    TArray<FSatAITrackedPiece> Pieces;
    if (!Builds.RemoveAndCopyValue(BuildId, Pieces))
    {
        return 0;
    }

    AFGLightweightBuildableSubsystem* Lightweight = AFGLightweightBuildableSubsystem::Get(GetWorld());
    int32 Removed = 0;
    for (int32 i = Pieces.Num() - 1; i >= 0; --i)
    {
        const FSatAITrackedPiece& Piece = Pieces[i];
        if (AActor* Actor = Piece.Actor.Get())
        {
            if (Actor->Implements<UFGDismantleInterface>())
            {  
                IFGDismantleInterface::Execute_PreUpgrade(Actor);
                IFGDismantleInterface::Execute_Dismantle(Actor); 
            }  
            else
            {
                Actor->Destroy();
            } 
            ++Removed;
            continue;
        }
        if (Piece.LightweightClass && Lightweight)
        {
            FRuntimeBuildableInstanceData* Data =
                Lightweight->GetRuntimeDataForBuildableClassAndIndex(Piece.LightweightClass, Piece.LightweightIndex);
            if (Data && Data->Handles.Num() > 0 && Data->Transform.GetLocation().Equals(Piece.Location, 1.0))
            {
                Lightweight->RemoveByInstanceIndex(Piece.LightweightClass, Piece.LightweightIndex);
                ++Removed;
            }
        }
    }
    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: cleared build '%s' (%d pieces)"), *BuildId, Removed);
    return Removed;
}

bool ASatAISubsystem::BuildPieces(const FString& BuildId, const TArray<FSatAIPiece>& Pieces,
    int32& OutBuilt, FString& OutError)
{
    OutBuilt = 0;
    if (Builds.Contains(BuildId))
    {
        OutError = FString::Printf(TEXT("build '%s' already exists; clear it first"), *BuildId);
        return false;
    }
    Builds.Add(BuildId);

    TMap<FString, AFGBuildable*> Machines;

    for (const FSatAIPiece& Piece : Pieces)
    {
        auto Fail = [&](const FString& Why)
            {
                OutError = FString::Printf(TEXT("piece '%s': %s (%d of %d built; clear '%s' to undo)"),
                    *Piece.Id, *Why, OutBuilt, Pieces.Num(), *BuildId);
                return false;
            };

        FString Error;
        if (Piece.Kind == TEXT("foundation") || Piece.Kind == TEXT("machine")
            || Piece.Kind == TEXT("attachment") || Piece.Kind == TEXT("container"))
        {
            FSatAISpawnResult Result;
            if (!SpawnTracked(BuildId, Piece.ClassPath, Piece.BuiltWith, Piece.Transform, Result, Error))
            {
                return Fail(Error);
            }
            Builds[BuildId].Last().Id = Piece.Id;
            if (Piece.Kind != TEXT("foundation"))
            {
                if (!Result.Actor)
                {
                    return Fail(TEXT("unexpectedly became lightweight"));
                }
                if (Piece.Kind == TEXT("machine") && !SetMachineRecipe(Result.Actor, Piece.Recipe, Error))
                {
                    return Fail(Error);
                }
                if (!Piece.FillItem.IsEmpty() && !FillContainer(Result.Actor, Piece.FillItem, Piece.FillAmount, Error))
                {
                    return Fail(Error);
                }
                Machines.Add(Piece.Id, Result.Actor);
            }
        }
        else if (Piece.Kind == TEXT("belt"))
        {
            AFGBuildable** From = Machines.Find(Piece.From);
            AFGBuildable** To = Machines.Find(Piece.To);
            if (!From || !To)
            {
                return Fail(TEXT("from/to must be machines, attachments or containers built earlier in this request"));
            }
            AFGBuildableConveyorBelt* Belt = ConnectWithBelt(*From, *To, Piece.ClassPath, Error,
                Piece.FromPort, Piece.ToPort);
            if (!Belt)
            {
                return Fail(Error);
            }
            Builds.FindOrAdd(BuildId).Add({ Belt, nullptr, INDEX_NONE, Belt->GetActorLocation(), Piece.Id });
        }
        else
        {
            return Fail(FString::Printf(TEXT("unknown kind '%s'"), *Piece.Kind));
        }
        ++OutBuilt;
    }

    UE_LOG(LogTemp, Warning, TEXT("SatAIBridge: built '%s' (%d pieces)"), *BuildId, OutBuilt);
    return true;
}

bool ASatAISubsystem::VerifyBuild(const FString& BuildId, TArray<FSatAIPieceReport>& Out, FString& OutError)
{
    const TArray<FSatAITrackedPiece>* Pieces = Builds.Find(BuildId);
    if (!Pieces)
    {
        OutError = FString::Printf(TEXT("no build '%s' (tracking is lost when the save is reloaded)"), *BuildId);
        return false;
    }

    TMap<const AActor*, FString> IdByActor;
    for (const FSatAITrackedPiece& Piece : *Pieces)
    {
        if (const AActor* Actor = Piece.Actor.Get())
        {
            IdByActor.Add(Actor, Piece.Id);
        }
    }

    AFGLightweightBuildableSubsystem* Lightweight = AFGLightweightBuildableSubsystem::Get(GetWorld());
    for (const FSatAITrackedPiece& Piece : *Pieces)
    {
        FSatAIPieceReport& Report = Out.AddDefaulted_GetRef();
        Report.Id = Piece.Id;

        if (Piece.LightweightClass)
        {
            Report.bLightweight = true;
            Report.ClassName = Piece.LightweightClass->GetName();
            FRuntimeBuildableInstanceData* Data = Lightweight
                ? Lightweight->GetRuntimeDataForBuildableClassAndIndex(Piece.LightweightClass, Piece.LightweightIndex)
                : nullptr;
            if (Data && Data->Handles.Num() > 0 && Data->Transform.GetLocation().Equals(Piece.Location, 1.0))
            {
                Report.bExists = true;
                Report.Transform = Data->Transform;
            }
            continue;
        }

        AActor* Actor = Piece.Actor.Get();
        if (!Actor)
        {
            continue;
        }
        Report.bExists = true;
        Report.ClassName = Actor->GetClass()->GetName();
        Report.Transform = Actor->GetActorTransform();

        TArray<UFGFactoryConnectionComponent*> Ports;
        Actor->GetComponents<UFGFactoryConnectionComponent>(Ports);
        for (UFGFactoryConnectionComponent* Port : Ports)
        {
            FSatAIPortReport& PortReport = Report.Ports.AddDefaulted_GetRef();
            PortReport.Name = Port->GetName();
            const EFactoryConnectionDirection Dir = Port->GetDirection();
            PortReport.Direction = Dir == EFactoryConnectionDirection::FCD_INPUT ? TEXT("in")
                : Dir == EFactoryConnectionDirection::FCD_OUTPUT ? TEXT("out") : TEXT("any");
            PortReport.bConnected = Port->IsConnected();

            if (UFGFactoryConnectionComponent* Other = Port->GetConnection())
            {
                const AActor* OtherActor = Other->GetOwner();
                const FString* OtherId = IdByActor.Find(OtherActor);
                PortReport.ConnectedTo = OtherId ? *OtherId : (OtherActor ? OtherActor->GetName() : FString());
            }
        }
    }
    return true;
}