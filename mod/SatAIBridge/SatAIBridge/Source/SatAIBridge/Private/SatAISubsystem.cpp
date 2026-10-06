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

AFGBuildable* ASatAISubsystem::SpawnBuildable(const FString& ClassPath, const FTransform& Transform, FString& OutError)
{
    UClass* BuildClass = LoadClass<AFGBuildable>(nullptr, *ClassPath);
    if (!BuildClass)
    {
        OutError = FString::Printf(TEXT("class not found: %s"), *ClassPath);
        return nullptr;
    }

    AFGBuildable* Buildable = GetWorld()->SpawnActorDeferred<AFGBuildable>(
        BuildClass, Transform, nullptr, nullptr,
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn);
    if (!Buildable)
    {
        OutError = TEXT("SpawnActorDeferred returned null");
        return nullptr;
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