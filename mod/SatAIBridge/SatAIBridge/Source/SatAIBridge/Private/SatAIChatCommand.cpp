#include "SatAIChatCommand.h"
#include "Command/CommandSender.h"

ASatAIChatCommand::ASatAIChatCommand()
{
    CommandName = TEXT("ai");
    MinNumberOfArguments = 1;
    Usage = NSLOCTEXT("SatAIBridge", "AIUsage", "/ai ping");
}

EExecutionStatus ASatAIChatCommand::ExecuteCommand_Implementation(
    UCommandSender* Sender, const TArray<FString>& Arguments, const FString& Label)
{
    const FString& Sub = Arguments[0];

    if (Sub.Equals(TEXT("ping"), ESearchCase::IgnoreCase))
    {
        Sender->SendChatMessage(TEXT("SatAIBridge: pong"));
        return EExecutionStatus::COMPLETED;
    }

    Sender->SendChatMessage(FString::Printf(TEXT("Unknown subcommand '%s'. Try: /ai ping"), *Sub));
    return EExecutionStatus::BAD_ARGUMENTS;
}