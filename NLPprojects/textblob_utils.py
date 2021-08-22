from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def get_subjectivity(text):
    """ calculate text subjectivity """
    return TextBlob(text).sentiment.subjectivity


def get_polarity(text):
    """ calculate text polarity """
    return TextBlob(text).sentiment.polarity


def get_analysis(score):
    """ calculate sentiment score """
    if score < 0:
        return "Negative"
    elif score > 0:
        return "Positive"
    else:
        return "Neutral"


def add_subjectivity_and_polarity(df, label="full_text"):
    """ add textblob subjectivity and polarity to dataframe """

    df["subjectivity"] = df[label].apply(get_subjectivity)
    df["polarity"] = df[label].apply(get_polarity)
    return df


def add_textblob_analysis(df, label="polarity"):
    """ add textblob analysis to dataframe """

    df["textblob_analysis"] = df[label].apply(get_analysis)
    return df


def plot_wordcloud(df, label="full_text"):
    all_words = " ".join([text for text in df[label]])
    wordcloud = WordCloud(width=800, height=400, max_font_size=120, background_color="white").generate(all_words)
    plt.figure(figsize=(18, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
