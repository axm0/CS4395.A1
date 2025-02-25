import math


def preprocess_text(text):
    """
    Tokenize the text using whitespace.
    If any additional preprocessing is needed, it can be added here
    Returns:
        tokens: each token is a single word or punctuation or ...
    """
    tokens = text.strip().split()

    return tokens


def build_counts(tokens, unk_threshold=1):
    """
    Build unigram and bigram counts.
    Any token with frequency <= unk_threshold, defaults tp 1, is replaced with an <UNK> token.
    Returns:
      unigram_counts: Counter for unigrams
      bigram_counts: Counter for bigrams
      vocab: set of tokens including <UNK>
    """
    # Count all occurrences of the tokens.
    init_count = {}
    for token in tokens:
        init_count[token] = init_count.get(token, 0) + 1
    # Build vocabulary words with frequency > unk_threshold
    vocab = {token for token, count in init_count.items() if count > unk_threshold}
    # Add <UNK> to vocab
    vocab.add("<UNK>")
    del init_count

    # Replace rare words (<= unk_threshold) with <UNK>
    processed_tokens = [token if token in vocab else "<UNK>" for token in tokens]

    # Recount occurrences using the processed_tokens
    # For Unigram
    unigram_counts = {}
    for token in processed_tokens:
        unigram_counts[token] = unigram_counts.get(token, 0) + 1
    # For Bigram
    bigram_counts = {}
    for i in range(len(processed_tokens) - 1):
        bigram = (processed_tokens[i], processed_tokens[i + 1])
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    return unigram_counts, bigram_counts, vocab


def compute_unigram_probabilities(unigram_counts):
    """
    Compute the unsmoothed unigram probabilities.
    Probability calculated as nb_of_word_occurrences / total_count
    Returns:
        A dictionary of the unigram probabilities
    """
    total_count = 0
    for count in unigram_counts.values():
        total_count += count

    return {word: count / total_count for word, count in unigram_counts.items()}


def compute_bigram_probabilities(unigram_counts, bigram_counts):
    """
    Compute unsmoothed bigram probabilities.
    Denote tokens word_1 and word_2 as w1 and w2 respectively.
    For each bigram (w1, w2), P(w2 | w1) = count(w1, w2) / count(w1)
    """
    bigram_probs = {}
    for (w1, w2), count in bigram_counts.items():
        bigram_probs[(w1, w2)] = count / unigram_counts[w1]

    return bigram_probs


def add_k_smoothing(unigram_counts, bigram_counts, vocab, k=1):
    """
    Compute bigram probabilities with add-k smoothing.
    Note that k=1 is the laplace smoothing
    Denote tokens word_1 and word_2 as w1 and w2 respectively.
    Denote vocab length as V.
    Denote k as the add-k factor.
    For each bigram (w1, w2), P(w2 | w1) = (count(w1, w2) + k) / (count(w1) + k * |V|)
    Returns:
        smoothed_probs:
    """
    V = len(vocab)
    smoothed_probs = {}
    for w1 in vocab:
        for w2 in vocab:
            count_bigram = bigram_counts.get((w1, w2), 0)
            count_unigram = unigram_counts.get(w1, 0)
            smoothed_probs[(w1, w2)] = (count_bigram + k) / (count_unigram + k * V)

    return smoothed_probs


def calculate_perplexity(tokens, unigram_probs=None, bigram_probs=None, model="bigram",
                         unigram_counts=None, vocab=None, k=1):
    """
    Calculate perplexity for the given sequence of tokens.
    For a unigram model, use unigram_probs.
    For a bigram model, use bigram_probs if provided; otherwise, perform on-the-fly add-k smoothing.
    A small epsilon (1e-10) is used for unseen n-grams.
    Denote tokens word_1 and word_2 as w1 and w2 respectively.
    Returns:
        perplexity
    """
    N = len(tokens)
    log_prob_sum = 0.0
    epsilon = 1e-10  # fallback probability for unseen n-grams

    if model == "unigram":
        for token in tokens:
            prob = unigram_probs.get(token, epsilon)
            log_prob_sum += math.log(prob)
    elif model == "bigram":
        for i in range(1, len(tokens)):
            w1 = tokens[i - 1]
            w2 = tokens[i]
            # Use provided bigram probabilities if available
            if bigram_probs is not None:
                prob = bigram_probs.get((w1, w2), epsilon)
            else:
                # On-the-fly add-k smoothing if bigram_probs is not precomputed
                count_bigram = bigram_counts.get((w1, w2), 0)
                count_unigram = unigram_counts.get(w1, 0)
                V = len(vocab)
                prob = (count_bigram + k) / count_unigram + k * V
            log_prob_sum += math.log(prob)
    avg_log_prob = log_prob_sum / N
    perplexity = math.exp(-avg_log_prob)
    return perplexity


if __name__ == "__main__":
    # Read training data (we know that each line is a review)
    with open("train.txt", "r") as f:
        train_data = f.read()
    train_lines = train_data.strip().split("\n")
    del train_data

    # Combine tokens from all reviews (all lines)
    train_tokens = []
    for line in train_lines:
        tokens = preprocess_text(line)
        train_tokens.extend(tokens)
    del train_lines

    # Build counts and vocabulary with unknown word handling (words with frequency <= unk_threshold (default=1) become <UNK>)
    unigram_counts, bigram_counts, vocab = build_counts(train_tokens, unk_threshold=1)
    del train_tokens

    # Compute unsmoothed probabilities
    unigram_probs = compute_unigram_probabilities(unigram_counts)
    bigram_probs = compute_bigram_probabilities(unigram_counts, bigram_counts)

    # Read validation data
    with open("val.txt", "r") as f:
        val_data = f.read()
    val_lines = val_data.strip().split("\n")

    # Process validation tokens.
    # Replace any word not in the training vocab with <UNK>
    val_tokens = []
    for line in val_lines:
        tokens = preprocess_text(line)
        tokens = [token if token in vocab else "<UNK>" for token in tokens]
        val_tokens.extend(tokens)
        del tokens

    # Compute smoothed bigram probabilities (using add-k smoothing)
    # Note that for k=1, that is the Laplace smoothing!
    k_list = [0.1, 0.5, 1, 1.5, 2, 5, 10]
    bigram_probs_smoothed = []
    for k in k_list:
        bigram_probs_smoothed.append(add_k_smoothing(unigram_counts, bigram_counts, vocab, k=k))

    # Calculate perplexity for different models:
    perplexity_unigram = calculate_perplexity(val_tokens, unigram_probs=unigram_probs, model="unigram")
    print("Validation Perplexity (Unigram):", perplexity_unigram)

    perplexity_bigram_unsmoothed = calculate_perplexity(val_tokens, bigram_probs=bigram_probs, model="bigram")
    print("Validation Perplexity (Bigram, Unsmooth):", perplexity_bigram_unsmoothed)

    for pos, item in enumerate(bigram_probs_smoothed):
        perplexity_bigram_smoothed = calculate_perplexity(val_tokens, bigram_probs=item, model="bigram")
        if k_list[pos] == 1:
            print("Validation Perplexity (Bigram, Laplace Smoothing, k=1):", perplexity_bigram_smoothed)
        else:
            print("Validation Perplexity (Bigram, Smoothed, k=" + str(k_list[pos]) + "):", perplexity_bigram_smoothed)

    del val_tokens, unigram_probs, bigram_probs, bigram_probs_smoothed
    del perplexity_unigram, perplexity_bigram_unsmoothed, perplexity_bigram_smoothed
