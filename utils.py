import tempfile
from collections import defaultdict

import pandas as pd
from pytubefix import YouTube


def load_dict(
    t2s_dict_file="./data/STCharacters.txt",
    jyutping_dict_file="./data/jyut6ping3.chars.dict.tsv",
    chars_freq_dict_file="./data/chars_freq.tsv",
):
    """Load Jyutping dictionary, character frequency dictionary, and traditional to simplified mapping.

    Args:
        t2s_dict_file (str): Path to the traditional to simplified mapping file.
        jyutping_dict_file (str): Path to the Jyutping dictionary file.
        chars_freq_dict_file (str): Path to the character frequency database file.

    Returns:
        tuple:
            traditional to simplified mapping,
            character to Jyutping dictionary,
            Jyutping to character dictionary
            character frequency dictionary.

    """

    # Load character frequencies
    chars_freq_df = pd.read_csv(chars_freq_dict_file, sep="\t", names=["char", "freq"])
    chars_freq = dict(zip(chars_freq_df.char, chars_freq_df.freq))

    # Load jyutping dictionary
    jyutping_df = pd.read_csv(jyutping_dict_file, sep="\t", names=["char", "jyutping"])

    # Create char_jyutping_dict
    char_jyutping_dict = defaultdict(list)
    for _, row in jyutping_df.iterrows():
        char_jyutping_dict[row["char"]].append(row["jyutping"])
    char_jyutping_dict = dict(char_jyutping_dict)

    # Create jyutping_char_dict
    jyutping_char_dict = defaultdict(list)
    for _, row in jyutping_df.iterrows():
        jyutping_char_dict[row["jyutping"]].append(row["char"])

    # Sort characters by frequency
    jyutping_char_dict = {
        k: sorted(v, key=lambda x: chars_freq.get(x, 0), reverse=True)
        for k, v in jyutping_char_dict.items()
    }

    # Load traditional to simplified mapping
    t2s_char_dict = {}
    t2s_df = pd.read_csv(t2s_dict_file, sep="\t", names=["sc", "tc"], encoding="utf-8")
    for _, row in t2s_df.iterrows():
        t2s_char_dict[row["sc"]] = row["tc"].split()

    # Add patches
    t2s_char_dict["晒"] = ["晒", "曬"]
    t2s_char_dict["咁"] = ["咁", "噉"]
    t2s_char_dict["旧"] = t2s_char_dict["旧"] + ["嚿"]

    return t2s_char_dict, char_jyutping_dict, jyutping_char_dict, chars_freq


def download_youtube_audio(video_id: str) -> str:
    """
    Download audio from YouTube video.

    Args:
        video_id (str): YouTube video ID.

    Returns:
        str: Path to the downloaded audio file.
    """
    urls = "https://www.youtube.com/watch?v={}".format(video_id)

    try:
        # https://github.com/JuanBindez/pytubefix/issues/242#issuecomment-2369067929
        vid = YouTube(urls, "MWEB")

        if vid.title is None:
            return None

        audio_download = vid.streams.get_audio_only()
        audio_download.download(
            mp3=True,
            filename=video_id,
            output_path=tempfile.gettempdir(),
            skip_existing=True,
        )
        audio_file = tempfile.gettempdir() + "/" + video_id + ".mp3"

        return audio_file

    except Exception as e:
        print(e)
        return None
