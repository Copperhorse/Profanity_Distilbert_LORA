# Profanity detection

## How we did it

While we didn't provide any fast api development, we did deliver the model as intended. Our process was divided into several steps.

-   **Step 1**: Training a profanity classifier:

    While the way examinee tackled the profanity was indeed very similar to my first attempt at this model, I quickly realized that making a black-list of words will simply not work. This is because context matters in case of language. (tits is a bird as well as female organ,) also because the language evolves along time and having a fixed list will make it tougher for the model evolve as well. To solve this. I first trained a distilbert classifier on the profanity data. This way the distilbert model is capable of learning the context and will make the model somewhat resistant to change in languages.

-   **Step 2**: Creating the dataset:

    After creating the classifier, The next step was to use that classifier to create synthetic dataset. I created the synthetic dataset using the amazon reviews dataset. In that dataset, the stars label is a good indicator of sentiment. (label <3 means negative sentiment and label >= 3 means positive sentiment). The problem would have been to find profanity. Since there is no definite indicator of profanity already existing in the dataset. As such, We first used the profanity classifier to predict/infer whether a certain review contained profanity or not. Then we created our synthetic dataset with 50 % profanity and 50% non profanity dataset.

    Here a question should arise if you were thinking like me, Why not train a sentiment classifier instead of a profanity classifier and had it infer the sentiment on the profanity dataset. The answer to that is the fact the model would be working on reviews dataset and not the profanity dataset in the wild. As such the training dataset should be very similar to the data present in the wild.

    The synthetic dataset had only the sentiment, the reviews and binary label whether it contained profanity or not

-   **Step 3** Finetuning:

    After creating the synthetic dataset, the next step was to fine-tune the model. Unlike the examinee which used distilbert only (unable to perform generative tasks) or what was said in the google doc, llama 3 8b, which is large and probably won't fit into gpu without quantization. I opted for LFM 2.5 1.2b, my reasoning being its ability to follow instruction, the underlying mamba architecture giving me long context length to work with while also maintaining a small size.
    We performed quantized fine-tuning instead of full training due to our small dataset.
    Now, for the people well versed in fine-tuning would know that the data required for fine-tuning must contain everything we want the model to do, however our current synthetic dataset doesn't have the generated positive reviews. The better option here would be to use a bigger LLM like chat gpt or claude to go through our dataset and generate reviews for the reviews containing profanity, Problem with this solution is that this requires money, money which we didn't have.
    So we will have to finetune(lora/qlora) the model on the synthetic data as is, But this leads to another problem. An llm trained on classification tasks stuggles with generation tasks, [Source](https://arxiv.org/html/2403.09162v1#S4). To overcome both of these problems, We first fine-tuned the llm on the classification task, And when it was required for the model to generate the alternative review, we would removed the lora adapter we had applied, essentially de-lobotomizing the model and it would be able to perform the tasks properly.

-   **Step 4** Inference:

    After we had fine-tuned the model, We provided code for inference that will help the client run the model. We also performed extensive testing on the model, giving it 12 reviews, divided into positive reviews with profanity, negative reviews with profanity, positive reviews with no profanity and finally negative reviews with no profanity.

### Source to the notebooks:

[Profanity classifier](https://colab.research.google.com/drive/1M9C1Mjn2s8i3OL-cbsTY_jylY9CNsibw?usp=sharing)

[Creating the dataset](https://colab.research.google.com/drive/1b1S1eTwg-pptfprx7YJR1fhuswaUUHv4?usp=sharing)

[Fine-tuning the dataset](https://colab.research.google.com/drive/1V3W6Obf8QVip-J9uTAKhSEbyEd7_CNSQ?usp=sharing)

[Inference and testing](https://colab.research.google.com/drive/1Bl1gI-QuDb7g6Hy8VksTi-zeRT7tsiWY?usp=sharing)

