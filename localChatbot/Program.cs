using LangChain.Memory;
using LangChain.Providers;
using LangChain.Providers.Ollama;
using localChatbot;
using static LangChain.Chains.Chain;

Console.WriteLine("Select from the following options:");

int choiceNumber = 1;

var choiceTexts = new[]
{
    nameof(ChatHistory),
    nameof(LocalRetrievalAugmentedGeneration),
    nameof(LocalRetrievalAugmentedGenerationWithHistory)
};
foreach (var choiceText in choiceTexts)
{
    Console.WriteLine($"    {choiceNumber}: {choiceText}");
    choiceNumber++;
}

Console.WriteLine();
Console.Write("Enter choice: ");

string choiceEntry = Console.ReadLine() ?? string.Empty;
if (int.TryParse(choiceEntry, out int choiceIndex))
{
    var choiceText = choiceTexts[--choiceIndex];
    
    Console.WriteLine();
    Console.WriteLine($"You selected '{choiceText}'");

    switch (choiceText)
    {
        case nameof(ChatHistory):
            await ChatHistory.StartConversation();
            break;
        case nameof(LocalRetrievalAugmentedGeneration):
            await LocalRetrievalAugmentedGeneration.StartConversation();
            break;
        case nameof(LocalRetrievalAugmentedGenerationWithHistory):
            await LocalRetrievalAugmentedGenerationWithHistory.StartConversation();
            break;
        default:
            throw new ArgumentOutOfRangeException($"Invalid choice index");
    }
}