﻿using Microsoft.Extensions.Configuration;

using OpenAI.Embeddings;

using OpenAI.Responses;



var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();

if (config["OPENAI_API_KEY"] == null)
{
    Console.WriteLine("Please set the OPENAI_API_KEY in your user secrets.");
    return;
}


var client = new EmbeddingClient("text-embedding-3-large", config["OPENAI_API_KEY"]);

 

var responses = await client.GenerateEmbeddingsAsync([
        "I have a dog, that is very cute",
        "War kills alot of people",
    ]);

 

var response1 = responses.Value[0].ToFloats();

var response2 = responses.Value[1].ToFloats();



 

var similarity_1_2 = DotProduct(response1, response2);

Console.WriteLine($"Similarity between response 1 and 2: {similarity_1_2}");



 

static float DotProduct(ReadOnlyMemory<float> a, ReadOnlyMemory<float> b)

{

    float sum = 0;

    for (int i = 0; i < a.Length; i++)

    {

        sum += a.Span[i] * b.Span[i];

    }

    return sum;

}