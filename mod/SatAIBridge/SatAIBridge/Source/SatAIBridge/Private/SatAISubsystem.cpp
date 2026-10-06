#include "SatAISubsystem.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "EngineUtils.h"  
#include "Buildables/FGBuildable.h"
#include "Buildables/FGBuildableManufacturer.h"
#include "FGRecipe.h"
#include "FGFactoryConnectionComponent.h"
#include "Buildables/FGBuildableConveyorBelt.h"
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
            return true;
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

UFGFactoryConnectionComponent* ASatAISubsystem::FindFreePort(AFGBuildable* Buildable, bool bOutput) const
{
    const EFactoryConnectionDirection Wanted =
        bOutput ? EFactoryConnectionDirection::FCD_OUTPUT : EFactoryConnectionDirection::FCD_INPUT;

    TArray<UFGFactoryConnectionComponent*> Ports;
    Buildable->GetComponents<UFGFactoryConnectionComponent>(Ports);
    for (UFGFactoryConnectionComponent* Port : Ports)
    {
        if (Port->GetDirection() == Wanted && !Port->IsConnected())
        {
            return Port;
        }
    }
    return nullptr;
}

AFGBuildableConveyorBelt* ASatAISubsystem::ConnectWithBelt(AFGBuildable* From, AFGBuildable* To,
    const FString& BeltClassPath, FString& OutError)
{
    UFGFactoryConnectionComponent* OutPort = FindFreePort(From, true);
    UFGFactoryConnectionComponent* InPort = FindFreePort(To, false);
    if (!OutPort || !InPort)
    {
        OutError = TEXT("no free output on source or no free input on target");
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
            Actor->Destroy();
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
        if (Piece.Kind == TEXT("foundation") || Piece.Kind == TEXT("machine"))
        {
            FSatAISpawnResult Result;
            if (!SpawnTracked(BuildId, Piece.ClassPath, Piece.BuiltWith, Piece.Transform, Result, Error))
            {
                return Fail(Error);
            }
            if (Piece.Kind == TEXT("machine"))
            {
                if (!Result.Actor)
                {
                    return Fail(TEXT("machine unexpectedly became lightweight"));
                }
                if (!SetMachineRecipe(Result.Actor, Piece.Recipe, Error))
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
                return Fail(TEXT("from/to must be machines built earlier in this request"));
            }
            AFGBuildableConveyorBelt* Belt = ConnectWithBelt(*From, *To, Piece.ClassPath, Error);
            if (!Belt)
            {
                return Fail(Error);
            }
            Builds.FindOrAdd(BuildId).Add({ Belt, nullptr, INDEX_NONE, Belt->GetActorLocation() });
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