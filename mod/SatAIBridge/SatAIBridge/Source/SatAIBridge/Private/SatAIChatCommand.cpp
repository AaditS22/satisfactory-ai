#include "SatAIChatCommand.h"
#include "Command/CommandSender.h"
#include "SatAISubsystem.h"

ASatAIChatCommand::ASatAIChatCommand()
{
    CommandName = TEXT("ai");
    MinNumberOfArguments = 1;
    Usage = NSLOCTEXT("SatAIBridge", "AIUsage", "/ai ping");
}

EExecutionStatus ASatAIChatCommand::ExecuteCommand_Implementation(
    UCommandSender* Sender, const TArray<FString>& Arguments, const FString& Label)
{
    const FString& SubCommand = Arguments[0];

    if (SubCommand == TEXT("ping"))
    {
        ASatAISubsystem* Subsystem = ASatAISubsystem::Get(this);
        if (!Subsystem)
        {
            Sender->SendChatMessage(TEXT("SatAIBridge: subsystem not found (registered in RootGameWorld?)"));
            return EExecutionStatus::UNCOMPLETED;
        }
        Sender->SendChatMessage(Subsystem->HandlePing());
        return EExecutionStatus::COMPLETED;
    }

    Sender->SendChatMessage(FString::Printf(TEXT("Unknown subcommand '%s'. Try: /ai ping"), *SubCommand));
    return EExecutionStatus::BAD_ARGUMENTS;
}