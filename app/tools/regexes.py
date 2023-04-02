import re


def remove_punctuation(input_string):
    # Define the regex pattern to match punctuation characters
    punctuation_pattern = r"[.,;`'\"!?-]"

    # Replace all matched punctuation characters with an empty string
    output_string = re.sub(punctuation_pattern, "", input_string)

    return output_string


def clean_positive_decimal(input_string):
    # Define the regex pattern to match punctuation characters
    punctuation_pattern = r"[;`'\"!?-]"

    # Replace all matched punctuation characters with an empty string
    output_string = re.sub(punctuation_pattern, "", input_string)

    return output_string
