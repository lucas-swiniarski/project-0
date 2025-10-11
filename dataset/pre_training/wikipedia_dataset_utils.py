import re
from datasets import Dataset

def coalesce_wikitext_articles(dataset_split):
    """
    Coalesce dataset of the same Wikipedia article into a single string.
    New articles are identified by titles like ' = {title} = \n'.
    """
    articles = []
    current_article_lines = []
    # Regex to find titles like ' = title = ' but not ' == subtitle == '
    title_pattern = re.compile(r'^\s=\s[^=].*[^=]\s=\s\n$')

    for item in dataset_split:
        line = item['text']
        # We check for titles that are not subtitles to delimit articles.
        # The first article does not start with a title, so we also check if current_article_lines is not empty.
        if title_pattern.match(line) and current_article_lines:
            articles.append("".join(current_article_lines).strip())
            current_article_lines = [line]
        else:
            # We only add non-empty lines to avoid too many blank lines inside articles.
            if line.strip():
                current_article_lines.append(line)

    # Add the last article
    if current_article_lines:
        articles.append("".join(current_article_lines).strip())

    return Dataset.from_dict({'text': articles})