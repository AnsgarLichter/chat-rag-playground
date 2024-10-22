using LangChain.Memory;
using LangChain.Providers;
using LangChain.Providers.Ollama;

using static LangChain.Chains.Chain;

namespace localChatbot;

public static class ChatHistory
{
    public static async Task StartConversation()
    {
        var provider = new OllamaProvider();
        var model = new OllamaChatModel(provider, id: "nemotron-mini");
        const string template = """

                                The following is a friendly conversation between a human and an AI.

                                {history}
                                Human: {input}
                                AI: 
                                """;
        
        var memory = PickMemoryStrategy(model);
        var chain =
            LoadMemory(memory, outputKey: "history")
            | Template(template)
            | LLM(model)
            | UpdateMemory(memory, requestKey: "input", responseKey: "text");

        Console.WriteLine();
        Console.WriteLine("Start a conversation with the friendly AI!");
        Console.WriteLine("(Enter 'exit' or hit Ctrl-C to end the conversation)");
        
        while (true)
        {
            Console.WriteLine();

            Console.Write("Human: ");
            var input = Console.ReadLine() ?? string.Empty;
            if (input == "exit")
            {
                break;
            }
            
            var currentChain = Set(input, "input")
                               | chain;
            var response = await currentChain.RunAsync("text");

            Console.Write("AI: ");
            Console.WriteLine(response);
        }
    }

    private static BaseChatMemory PickMemoryStrategy(IChatModel model)
    {
        var messageFormatter = new MessageFormatter
        {
            AiPrefix = "AI",
            HumanPrefix = "Human"
        };
        var chatHistory = GetChatMessageHistory();
        var memoryClassName = PromptForChoice(new[]
        {
            nameof(ConversationBufferMemory),
            nameof(ConversationWindowBufferMemory)
        });

        return memoryClassName switch
        {
            nameof(ConversationBufferMemory) => GetConversationBufferMemory(chatHistory, messageFormatter),
            nameof(ConversationWindowBufferMemory) => GetConversationWindowBufferMemory(chatHistory, messageFormatter),
            _ => throw new InvalidOperationException($"Unexpected memory class name: '{memoryClassName}'")
        };
    }

    private static string PromptForChoice(string[] choiceTexts)
    {
        while (true)
        {
            Console.Clear();
            Console.WriteLine("Select from the following options:");

            var choiceNumber = 1;
            foreach (var choiceText in choiceTexts)
            {
                Console.WriteLine($"    {choiceNumber}: {choiceText}");
                choiceNumber++;
            }

            Console.WriteLine();
            Console.Write("Enter choice: ");

            var choiceEntry = Console.ReadLine() ?? string.Empty;
            if (int.TryParse(choiceEntry, out int choiceIndex))
            {
                string choiceText = choiceTexts[choiceIndex];

                Console.WriteLine();
                Console.WriteLine($"You selected '{choiceText}'");

                return choiceText;
            }
        }
    }

    private static BaseChatMessageHistory GetChatMessageHistory()
    {
        return new ChatMessageHistory();
    }

    private static BaseChatMemory GetConversationBufferMemory(BaseChatMessageHistory chatHistory, MessageFormatter messageFormatter)
    {
        return new ConversationBufferMemory(chatHistory)
        {
            Formatter = messageFormatter
        };
    }

    private static BaseChatMemory GetConversationWindowBufferMemory(BaseChatMessageHistory chatHistory, MessageFormatter messageFormatter)
    {
        return new ConversationWindowBufferMemory(chatHistory)
        {
            WindowSize = 3,
            Formatter = messageFormatter
        };
    }
}