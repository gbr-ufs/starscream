# /// script
# requires-python = "==3.12"
# dependencies = [
#     "openai-whisper==20250625",
#     "polars==1.38.1",
#     "pyarrow==23.0.0",
#     "rich==14.3.2",
#     "seaborn==0.13.2",
#     "transformers==5.1.0",
#     "typer==0.21.1",
# ]
# ///


import mimetypes

from typing import Any, Callable

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import typer
import whisper

from transformers import pipeline, Pipeline
from whisper import Whisper


def get_mime_type(file_path: str) -> str:
    """Get the first identifier of a file's MIME type.

    Args:
        file_path: The path to the file whose type will be checked.

    Returns:
       The primary MIME type category (viz. 'audio', 'text'). 'unknown' if
       it cannot be identified.
    """

    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        return mime_type.split("/")[0]

    return "unknown"


def handle_audio(file_path: str) -> str:
    """Transcribe audio files through Whisper.

    Args:
        file_path: The path to the audio file whose contents will be
        transcribed.

    Returns:
        The transcribed text.
    """

    model: Whisper = whisper.load_model("medium")
    result = model.transcribe(file_path)

    return result["text"]


def handle_text(file_path: str) -> str:
    """Read the contents of a text file into a string.

    Args:
        file_path: The path to the text file whose contents will be read
        into a string.

    Returns:
       A string with the contents of the file.
    """

    with open(file_path, "r") as f:
        return f.read()


def analyze_sentiment(text: str) -> list[dict[str, Any]]:
    """Analyze text using DistilBERT. The model has three categories for text:
    - 'positive'
    - 'neutral'
    - 'negative'

    Args:
        text: The text to analyze.

    Returns:
        A list of dictionaries with keys 'label' and 'score'.
    """
    model: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

    sentiment_pipeline: Pipeline = pipeline(model=model, task="sentiment-analysis")

    # top_k=3 is used to force the pipeline to return all results.
    return sentiment_pipeline(text, top_k=3)


def plot(
    sentiment_data: list[dict[str, Any]], output: str, show: bool, title: str
) -> None:
    """Generate a bar chart from the result of the sentiment analysis process.

    Args:
    sentiment_data: The result of the sentiment analysis
    process done by the model.
    output: File path where the generated chart will be saved. If
    empty, the chart will not be saved.
    show: Whether to show the chart on the screen, if the environment
    (viz. Jupyter Notebook) permits.
    title: The title of the bar chart.
    """
    df = pl.DataFrame(sentiment_data)
    barplot = sns.barplot(
        data=df, x="label", y="score", hue="label", palette="flare", legend=False
    )

    sns.set_theme(style="ticks")

    plt.title(title)

    barplot.set(xlabel=None, ylabel=None)

    for i in barplot.containers:
        barplot.bar_label(i, fmt="%.2f")

    if output:
        plt.savefig(output)

    if show:
        plt.show()


def run(file_path: str, output: str, show: bool, title: str) -> None:
    """Main logic of the program. For the details on the arguments, see 'main'."""
    file_type_handler: dict[str, Callable[[str], str]] = {
        "audio": handle_audio,
        "text": handle_text,
    }
    file_type: str = get_mime_type(file_path)
    handler = file_type_handler.get(file_type)

    if not handler:
        print(f"Unsupported file type: {file_type}")

        raise typer.Exit(1)

    text: str = handler(file_path)
    sentiment_data = analyze_sentiment(text)

    if output or show:
        plot(sentiment_data, output, show, title)
    else:
        for data in sentiment_data:
            print(f"{data['label']}: {data['score']:.2f}")


def main(
    file_path: str = typer.Argument(..., help="Path to the audio or text file"),
    output: str = typer.Option("", help="File path to save the plot"),
    show: bool = typer.Option(
        False,
        help="Whether to display the plot, if the environment supports it (viz. Jupyter Notebook).",
    ),
    title: str = typer.Option("Sentiment Analysis Scores"),
) -> None:
    """
    Perform sentiment analysis on files.
    """

    run(file_path, output, show, title)


if __name__ == "__main__":
    typer.run(main)
