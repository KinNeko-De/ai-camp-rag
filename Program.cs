

﻿using System.ComponentModel.DataAnnotations;

using Microsoft.Extensions.Configuration;

using OpenAI.Embeddings;

using OpenAI.Responses;

 

var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();

 

var client = new EmbeddingClient("text-embedding-3-large", config["OPENAI_API_KEY"]);

 

string[] funnyAnimals =

[

    "Blurflebeast: A rainbow-striped lizard that sneezes confetti and only eats left shoes.",

    "Snoozlephant: A miniature elephant that sleeps 23 hours a day and snores like a tuba.",

    "Wobbleduck: A duck with three legs, two of which are used solely for interpretive dance.",

    "Fuzzlefox: A fox with cotton candy fur that attracts bees but repels socks.",

    "Moozard: A cow-sized lizard that gives chocolate milk when scared.",

    "Quibblepuff: A hamster that inflates like a balloon when startled and floats gently to the ceiling.",

    "Gigglegator: An alligator that laughs every time it hears the word 'pickle.'",

    "Twinkletoad: A toad with disco ball skin and a love for spontaneous breakdancing.",

    "Snorflecat: A cat with a trunk like an elephant and an uncontrollable urge to vacuum up spaghetti.",

    "Ploparoo: A kangaroo that can only jump backward and communicates through interpretive jazz hands.",

    "Glittercorn: A unicorn that leaves a trail of glitter wherever it goes, tells terrible puns, and can only grant wishes on Tuesdays."

];

 

var responses = await client.GenerateEmbeddingsAsync(funnyAnimals);

var vectorDatabase = responses.Value.Select(e => e.ToFloats()).ToArray();

 

var userInput = "How many animals do we have in our database?";

var userEmbedding = (await client.GenerateEmbeddingAsync(userInput)).Value.ToFloats();

 

var sortedBySimilarity = vectorDatabase

    .Select((vector, index) => new { Index = index, Similarity = DotProduct(userEmbedding, vector) })

    .OrderByDescending(x => x.Similarity)

    .Take(3)

    .ToArray();

 

var responseClient = new OpenAIResponseClient("gpt-4.1", config["OPENAI_API_KEY"]);

Console.WriteLine("----");

var systemPrompt = $"""

    You are an assistant that helps users answering questions about secret animals. Nobody

    knows about these animals, except we in our company. Only use the information

    provided in the following list of animals to answer the question. If you don't know

    the answer, say you don't know. Never make up information. Always answer based

    on the information provided in the list of animals.

 

    <Animals>

    {"<Animal>" + string.Join("</Animal>\n<Animal>", sortedBySimilarity.Select(x => funnyAnimals[x.Index])) + "</Animal>"}

    </Animals>

    """;

Console.WriteLine(systemPrompt);

 

var response = await responseClient.CreateResponseAsync(userInput, new()

{

    Instructions = systemPrompt

});

 

Console.WriteLine("----");

Console.WriteLine(response.Value.GetOutputText());

 

 

static float DotProduct(ReadOnlyMemory<float> a, ReadOnlyMemory<float> b)

{

    float sum = 0;

    for (int i = 0; i < a.Length; i++)

    {

        sum += a.Span[i] * b.Span[i];

    }

    return sum;

}