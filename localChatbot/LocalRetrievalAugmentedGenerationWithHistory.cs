using LangChain.Chains;
using LangChain.Databases.Sqlite;
using LangChain.DocumentLoaders;
using LangChain.Extensions;
using LangChain.Memory;
using LangChain.Providers.Ollama;

namespace localChatbot;

public static class LocalRetrievalAugmentedGenerationWithHistory
{
    public static async Task StartConversation()
    {
        var provider = new OllamaProvider();
        var vectorDatabase = new SqLiteVectorDatabase(dataSource: "vectors.db");
        var embeddingsModel = new OllamaEmbeddingModel(provider, "all-minilm");
        var chatModel = new OllamaChatModel(provider, id: "llama3");

        const string template = $$"""
                                  Use the following pieces of context to answer the question from the human at the end.
                                  If the answer is not in context then just say that you don't know, don't try to make up an answer.
                                  Keep the answer as short as possible.
                                  Context: {context}
                                  
                                  The following is a friendly conversation between a human and an AI.
                                  
                                  {history}
                                  Human: {question}
                                  AI: 
                                  """;
        var dataSource = DataSource.FromUrl(
            "https://www.fia.com/sites/default/files/decision-document/2024%20United%20States%20Grand%20Prix%20-%20Provisional%20Race%20Classification.pdf");
        var vectorCollection = await vectorDatabase.AddDocumentsFromAsync<PdfPigPdfLoader>(
            embeddingsModel,
            dimensions: 384, // Should be 384 for all-minilm
            dataSource: dataSource,
            collectionName: "f1Austin",
            textSplitter: null,
            behavior: AddDocumentsToDatabaseBehavior.JustReturnCollectionIfCollectionIsAlreadyExists);
        
        var memory = new ConversationBufferMemory(new ChatMessageHistory())
        {
            Formatter = new MessageFormatter
            {
                AiPrefix = "AI",
                HumanPrefix = "Human"
            }
        };
        var chain = Chain.LoadMemory(memory, outputKey: "history")
                    | Chain.Template(template, "prompt")
                    | Chain.LLM(chatModel, inputKey: "prompt", outputKey: "text")
                    | Chain.UpdateMemory(memory, requestKey: "question", responseKey: "text");
        
        Console.WriteLine();
        Console.WriteLine("Start a conversation with the friendly AI!");
        Console.WriteLine("(Enter 'exit' or hit Ctrl-C to end the conversation)");
        
        while (true)
        {
            Console.WriteLine();

            Console.Write("Human: ");
            var question = Console.ReadLine() ?? string.Empty;
            if (question == "exit")
            {
                break;
            }
            
            var currentChain = Chain.Set(question, "question")
                               | Chain.RetrieveDocuments(vectorCollection, embeddingsModel, 10, "question", "documents")
                               | Chain.StuffDocuments("documents", "context")
                               | chain;
            var response = await currentChain.RunAsync("text");

            Console.Write("AI: ");
            Console.WriteLine(response);
        }
    }
}