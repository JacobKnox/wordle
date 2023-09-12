import numpy as np
import os
from scipy import spatial
import tensorflow_hub as hub
import ssl

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'.upper()
ssl._create_default_https_context = ssl._create_unverified_context
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def main():
    pass

def makeguess(wordlist: list[str], guesses: list[str] = [], feedback: list[list[int]] = []) -> str:
    """
    Guess a word from the available wordlist, (optionally) using feedback
    from previous guesses.
    
    Parameters
    ----------
    wordlist : list of str
    A list of the valid word choices. The output must come from this list.
    guesses : list of str
    A list of the previously guessed words, in the order they were made,
    e.g. guesses[0] = first guess, guesses[1] = second guess. The length
    of the list equals the number of guesses made so far. An empty list
    (default) implies no guesses have been made.
    feedback : list of lists of int
    A list comprising one list per word guess and one integer per letter
    in that word, to indicate if the letter is correct (2), almost
    correct (1), or incorrect (0). An empty list (default) implies no
    guesses have been made.
    
    Output
    ------
    word : str
    The word chosen by the AI for the next guess.
    """
    
    # things to do on the first guess
    if len(guesses) == 0:
        # delete old save if it exists
        if 'syntaxscholars_pruned_list.npy' in os.listdir():
            os.remove('syntaxscholars_pruned_list.npy')
        # hardcode first answer
        return "TRACE"
    # if there is feedback, then we need to apply it
    if len(feedback) > 0:
        # if there is already a pruned list from the previous guesses, then load it
        if 'syntaxscholars_pruned_list.npy'  in os.listdir():
            our_loaded_word_list = np.load('syntaxscholars_pruned_list.npy')
        # otherwise, the wordlist is the list to go on
        else:
            our_loaded_word_list = wordlist
        pruned_word_list = []
        # get the most recent guess
        comparison_word = guesses[-1]
        # copy/convert it to a numpy array of characters for checking acceptable doubles
        comparison_word_copy = np.array(list(comparison_word))
        # get the most recent feedback
        comparison_info = feedback[-1]
        # find the acceptable doubles (letters that are yellow or green)
        acceptable_doubles = comparison_word_copy[np.where(np.array(comparison_info) > 0, True, False)].tolist()
        # loop over all the words in the list
        for word in our_loaded_word_list:
            flag = True
            for i in range(5):
                # if the letter is green, then make sure the word contains it in the same position
                if comparison_info[i] == 2:
                    if word[i] != comparison_word[i]:
                        flag = False
                        break
                # if the letter is yellow, then check if the word has it in the same place or doesn't have it at all
                elif comparison_info[i] == 1:
                    if word[i] == comparison_word[i] or not comparison_word[i] in word:
                        flag = False
                        break
                # else, the letter is gray, so check if it is in the same place OR if it's in the word AND not an acceptable double
                else:
                    if word[i] == comparison_word[i] or (comparison_word[i] in word and comparison_word[i] not in acceptable_doubles):
                        flag = False
                        break
            # if the flag is still True, then it meets all the conditions and should be added to the pruned list
            if flag:
                pruned_word_list.append(word)
        wordlist = pruned_word_list
        np.save('syntaxscholars_pruned_list.npy', pruned_word_list)
    # spatial counting thing
    ranks_per_spot = find_common_chars_positionally(wordlist)
    # create the "ideal" word
    ideal_word = "".join([spot[-1][0] for spot in ranks_per_spot])
    # do the distance thing and find the distances from the "ideal" word
    best_words = vector_analysis(ideal_word, wordlist)
    # return the word with the smallest distance
    return best_words[0][0].upper() # talk to doctor eicholtz

def vector_analysis(ideal_word: str, wordlist: list[str]) -> list[list]:
    """
    Vectorizes the wordlist and computes their distance from the "ideal"
    word.
    
    Parameters
    ----------
    ideal_word : str
    The theoretical "ideal" word based on frequency of letters in each
    position in the wordlist.
    wordlist : list of str
    A list of the valid word choices.
    
    Output
    ------
    distances_sorted : list of lists of format [str, int]
    A list of each word and their distance from the "ideal" word.
    """
    
    # vectorize words
    embeddings = np.array(embed(wordlist))
    ideal_word_embed = embed([ideal_word])[0]
    distances = []
    for i in range(len(embeddings)):
        # computer the distance from the ith vector to the ideal word and append it
        distance = spatial.distance.cosine(ideal_word_embed, embeddings[i])
        distances.append([wordlist[i], distance])
    # convert the distances to an np array, sort it, and convert it back to a list
    distances = np.array(distances)
    distances_sorted = distances[np.argsort(distances[:, 1])].tolist()
    # return the sorted distances
    return distances_sorted

def find_common_chars_positionally(words: list[str]):
    ranks = [] # ranks[i] = the ith position in a generic guess = [[char, occurances ], [a,b ], ...]
    # loop over each index
    for place in range(5):
        # initialize the character mapping
        mapping = dict(zip(list(ALPHABET), [0 for _ in range(len(ALPHABET))]))
        # loop over all words, look at the character at the ith index, and add 1 to the corresponding character count
        for word in words[1:]:
            ch = word[place]
            mapping[ch] += 1
        # append the resulting map to the ranks list
        ranks.append(sorted(mapping.items(), key=lambda x:x[1])) # sorted() time complexity is nlogn
    return ranks

if __name__ == "__main__":
    main()
