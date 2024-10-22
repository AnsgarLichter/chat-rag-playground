using LangChain.Databases.Sqlite;
using LangChain.DocumentLoaders;
using LangChain.Providers.Ollama;
using LangChain.Extensions;
using LangChain.Memory;
using Ollama;
using Message = LangChain.Providers.Message;
using MessageRole = LangChain.Providers.MessageRole;

var provider = new OllamaProvider();
//Converts text to embeddings
var embeddingModel = new OllamaEmbeddingModel(provider, "all-minilm");
var chatModel = new OllamaChatModel(provider, id: "nemotron-mini");
var chatSettings = new OllamaChatSettings
{
    UseStreaming = true
};

var vectorDatabase = new SqLiteVectorDatabase(dataSource: "vectors.db");

var vectorCollection = await vectorDatabase.AddDocumentsFromAsync<PdfPigPdfLoader>(
    embeddingModel,
    dimensions: 384, // Should be 384 for all-minilm
    dataSource: DataSource.FromUrl(
        "https://canonburyprimaryschool.co.uk/wp-content/uploads/2016/01/Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone-EnglishOnlineClub.com_.pdf"),
    collectionName: "harrypotter", // Use only if you want to have multiple collections
    textSplitter: null,
    behavior: AddDocumentsToDatabaseBehavior.JustReturnCollectionIfCollectionIsAlreadyExists);

chatModel.DeltaReceived += (_, delta) => Console.Write(delta.Content);
while (true)
{
    Console.WriteLine("Question: ");
    var question = Console.ReadLine();
    while (string.IsNullOrWhiteSpace(question))
    {
        Console.WriteLine("Please enter a question, not only whitespace: ");
        question = Console.ReadLine();
    }
    
    var similarDocuments = await vectorCollection.GetSimilarDocuments(embeddingModel, question, amount: 5);
    
    Console.WriteLine("Answer:");
    var request = $"""
                   Use the following pieces of context to answer the question at the end.
                   If the answer is not in context then just say that you don't know, don't try to make up an answer.
                   Keep the answer as short as possible.

                   {similarDocuments.AsString()}

                   Question: {question}
                   Helpful Answer:
                   """;
    await chatModel.GenerateAsync(
        request,
        chatSettings);
    Console.WriteLine();
}