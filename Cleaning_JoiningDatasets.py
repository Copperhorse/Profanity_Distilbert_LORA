## this code is used to combine and clean the dataset that will be used to finetune llms:
#  it will use the following datasets: https://huggingface.co/datasets/mteb/amazon_reviews_multi/tree/main/en
# https://huggingface.co/datasets/tarekziade/pardonmyai-clean
import re

import pandas as pd

# First we will create a set or list that will contain profanities, we will use this list to filter reviews that contain profanities:
# the list is from the following repo: https://github.com/snguyenthanh/better_profanity/blob/master/better_profanity/profanity_wordlist.txt
Profanities = {
    # Original
    "crap",
    "fuck",
    "shit",
    "f**k",
    "f*ck",
    "hoe",
    "bitch",
    "dick",
    "sh*t",
    "B*tch",
    # Common profanities
    "damn",
    "hell",
    "ass",
    "asshole",
    "bastard",
    "piss",
    "cunt",
    # Variations and related terms
    "fucking",
    "fucked",
    "fucker",
    "fucks",
    "motherfucker",
    "motherfucking",
    "shitty",
    "shitting",
    "bullshit",
    "horseshit",
    "shits",
    "bitches",
    "bitching",
    "bitchy",
    "dicks",
    "dickhead",
    "dumbass",
    "jackass",
    "pissed",
    "pissing",
    # Slang variations
    "wtf",
    "stfu",
    "gtfo",
    "omfg",
    "fml",
    # Censored versions (common patterns)
    "f***",
    "s**t",
    "b*tch",
    "d*ck",
    "a**",
    "a**hole",
    "f@ck",
    "sh!t",
    "b!tch",
    "a$$",
    "fuk",
    "fck",
    # Derogatory terms
    "whore",
    "slut",
    "skank",
    "twat",
    "douche",
    "douchebag",
    # Offensive slurs (be careful with context)
    "retard",
    "retarded",
    "fag",
    "faggot",
    "dyke",
    # Body parts used offensively
    "cock",
    "pussy",
    "boobs",
    "tits",
    # Other common offensive terms
    "sucked",
    "crappy",
    "bollocks",
    "wanker",
    "tosser",
    "bugger",
    "bloody",
    "blimey",
    # Insults
    "pathetic",
}

# Load and combine Amazon splits
amazon_train = pd.read_json("Dataset/Amazon/train.jsonl", lines=True)
amazon_test = pd.read_json("Dataset/Amazon/test.jsonl", lines=True)
amazon_validation = pd.read_json("Dataset/Amazon/validation.jsonl", lines=True)
amazon = pd.concat([amazon_train, amazon_test, amazon_validation], ignore_index=True)

# Strategy: Add spaces around the text, then match with spaces around profanity
# This prevents substring matches like "ass" in "glass"

# Step 1: Create a padded version with spaces
amazon["padded_text"] = " " + amazon["text"].str.lower() + " "

# Step 2: Build patterns with spaces (match " word ")
spaced_profanities = [f" {word.lower()} " for word in Profanities]
pattern_str = "|".join(re.escape(word) for word in spaced_profanities)

# Step 3: Fast filtering using native pandas operations
amazon["profanity"] = amazon["padded_text"].str.contains(
    pattern_str, regex=True, na=False
)

# Filter profane and non-profane rows
filtered_df = amazon[amazon["profanity"] == True]
non_profane_df = amazon[amazon["profanity"] == False]
filtered_df.to_json("profanity.json", orient="records", lines=True)

# Sample
sample_size = 3000
actual_size = min(sample_size, len(filtered_df), len(non_profane_df))
balanced_profane = filtered_df.sample(n=actual_size, random_state=42)
balanced_non_profane = non_profane_df.sample(n=actual_size, random_state=42)

# %% Cell 1
balanced_amazon = pd.concat([balanced_profane, balanced_non_profane], ignore_index=True)
balanced_amazon = balanced_amazon.sample(frac=1.0, random_state=42).reset_index(
    drop=True
)


# Step 4: Extract the profanity word (only for profane rows)
def extract_profanity_word(text):
    """Extract first matching profanity - only called on profane rows"""
    text_lower = f" {text.lower()} "
    for word in Profanities:
        if f" {word.lower()} " in text_lower:
            return word.lower()
    return None


# Only extract from profane rows to save time
profane_rows = balanced_amazon[balanced_amazon["profanity"] == True].copy()
non_profane_rows = balanced_amazon[balanced_amazon["profanity"] == False].copy()

# Extract profanity words from profane rows
profane_rows["profanity_word"] = profane_rows["text"].apply(extract_profanity_word)

# Add null profanity_word to non-profane rows
non_profane_rows["profanity_word"] = None

# Combine back
balanced_amazon = pd.concat([profane_rows, non_profane_rows], ignore_index=True)

# Drop the padded_text helper column
balanced_amazon = balanced_amazon.drop("padded_text", axis=1)


# Add sentiment category for easier analysis
# Labels: 0-1 = negative, 2 = neutral, 3-4 = positive
def categorize_sentiment(label):
    if label <= 1:
        return "negative"
    elif label == 2:
        return "neutral"
    else:  # label >= 3
        return "positive"


balanced_amazon["sentiment"] = balanced_amazon["label"].apply(categorize_sentiment)

# Save
# balanced_amazon.to_parquet("balanced_amazon.parquet", index=False)
balanced_amazon.to_csv("balanced_amazon.csv", index=False)

# Stats
print(balanced_amazon.shape)
print(balanced_amazon.groupby("profanity").size())
print(balanced_amazon.groupby("profanity_word").size().sort_values(ascending=False))

# Quick stats
print(balanced_amazon.shape)
print(balanced_amazon.groupby("profanity").size())
print(balanced_amazon.groupby(["label", "profanity"]).size().sort_index())
print("\nBy sentiment category:")
print(balanced_amazon.groupby(["sentiment", "profanity"]).size().sort_index())

# Show profanity word distribution
print("\nProfanity word counts:")
print(balanced_amazon.groupby("profanity_word").size().sort_values(ascending=False))

# Get profanity word frequency (excluding nulls)
profanity_freq = (
    balanced_amazon[balanced_amazon["profanity_word"].notna()]
    .groupby("profanity_word")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("Most common profanities:")
print(profanity_freq.head(20))

print("\nLeast common profanities (candidates for removal):")
print(profanity_freq.tail(20))

# Words appearing less than N times could be dropped
threshold = 10
rare_words = profanity_freq[profanity_freq["count"] < threshold]
print(f"\nWords appearing less than {threshold} times:")
print(rare_words)

# Analyze by sentiment label (0-1: negative, 2: neutral, 3-4: positive)
false_positive_analysis = (
    balanced_amazon[balanced_amazon["profanity_word"].notna()]
    .groupby(["profanity_word", "label"])
    .size()
    .reset_index(name="count")
    .sort_values(["profanity_word", "label"])
)

print("\nProfanity words by sentiment label:")
print(false_positive_analysis)

# Analyze by sentiment category
sentiment_analysis = (
    balanced_amazon[balanced_amazon["profanity_word"].notna()]
    .groupby(["profanity_word", "sentiment"])
    .size()
    .reset_index(name="count")
    .sort_values(["profanity_word", "sentiment"])
)

print("\nProfanity words by sentiment category:")
print(sentiment_analysis)

# Find words that appear disproportionately in positive reviews (potential false positives)
# Positive = labels 3-4
positive_profanity = (
    balanced_amazon[
        (balanced_amazon["profanity_word"].notna())
        & (balanced_amazon["label"] >= 3)  # Labels 3-4 are positive
    ]
    .groupby("profanity_word")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(
    "\nProfanities appearing in positive reviews (labels 3-4, potential false positives):"
)
print(positive_profanity.head(20))

# Also check neutral reviews (label 2)
neutral_profanity = (
    balanced_amazon[
        (balanced_amazon["profanity_word"].notna())
        & (balanced_amazon["label"] == 2)  # Label 2 is neutral
    ]
    .groupby("profanity_word")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\nProfanities appearing in neutral reviews (label 2):")
print(neutral_profanity.head(20))
