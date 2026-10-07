#pragma once

#include "CoreMinimal.h"
#include "Subsystem/ModSubsystem.h"
#include "HttpRouteHandle.h"
#include "SatAISubsystem.generated.h"

class AFGBuildable;
class UFGFactoryConnectionComponent;
class AFGBuildableConveyorBelt;
class IHttpRouter;

struct FSatAIPortGeometry
{
    FString Name;
    FString Direction;
    FVector LocalPosCm;
    FVector LocalFacing;
};

struct FSatAIClearanceBox
{
    FString Type;
    FBox LocalBox;
};

struct FSatAISpawnResult
{
    FString Name;
    FTransform Transform;
    FBox LocalBounds = FBox(ForceInit);
    bool bLightweight = false;
    AFGBuildable* Actor = nullptr;
    TArray<FSatAIPortGeometry> Ports;
    TArray<FSatAIClearanceBox> Clearance;
};

struct FSatAITrackedPiece
{
    TWeakObjectPtr<AActor> Actor;
    UClass* LightweightClass = nullptr;
    int32 LightweightIndex = INDEX_NONE;
    FVector Location = FVector::ZeroVector;
    FString Id;
};

struct FSatAIPiece
{
    FString Id;
    FString Kind;
    FString ClassPath;
    FString BuiltWith;
    FString Recipe;
    FString From, To;
    FString FromPort, ToPort;
    FString FillItem;   
    int32 FillAmount = 0;
    FTransform Transform;
};

struct FSatAIPortReport
{
    FString Name;
    FString Direction;
    bool bConnected = false;
    FString ConnectedTo;
};

struct FSatAIPieceReport
{
    FString Id;
    FString ClassName;
    bool bExists = false;
    bool bLightweight = false;
    FTransform Transform;
    TArray<FSatAIPortReport> Ports;
};

UCLASS(Abstract, Blueprintable)
class SATAIBRIDGE_API ASatAISubsystem : public AModSubsystem
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    static ASatAISubsystem* Get(const UObject* WorldContext);
    AFGBuildable* SpawnBuildable(const FString& ClassPath, const FTransform& Transform, FString& OutError,
        const FString& BuiltWithRecipePath = FString());
    bool SetMachineRecipe(AFGBuildable* Buildable, const FString& RecipePath, FString& OutError);
    void LogPorts(AFGBuildable* Buildable) const;
    UFGFactoryConnectionComponent* FindFreePort(AFGBuildable* Buildable, bool bOutput,
        const FString& PortName = FString()) const;
    AFGBuildableConveyorBelt* ConnectWithBelt(AFGBuildable* From, AFGBuildable* To,
        const FString& BeltClassPath, FString& OutError,
        const FString& FromPort = FString(), const FString& ToPort = FString());

    FString HandlePing();

    bool SpawnTracked(const FString& BuildId, const FString& ClassPath, const FString& BuiltWithRecipePath,
        const FTransform& Transform, FSatAISpawnResult& Out, FString& OutError);
    int32 ClearBuild(const FString& BuildId);
    bool BuildPieces(const FString& BuildId, const TArray<FSatAIPiece>& Pieces, int32& OutBuilt, FString& OutError);
    bool VerifyBuild(const FString& BuildId, TArray<FSatAIPieceReport>& Out, FString& OutError);

private:
    int32 FindLightweightIndex(UClass* Class, const FVector& LocationCm) const;

    void StartHttpServer();
    void StopHttpServer();

    static constexpr uint32 HttpPort = 18642;
    TSharedPtr<IHttpRouter> HttpRouter;
    TArray<FHttpRouteHandle> HttpRoutes;

    TMap<FString, TArray<FSatAITrackedPiece>> Builds;

    int32 PingCount = 0;
    static TWeakObjectPtr<ASatAISubsystem> Instance;
};