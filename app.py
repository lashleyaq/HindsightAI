import streamlit as st
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import os
import plotly.express as px

# ========================#
st.set_page_config(
    page_title="HindsightAI",
    page_icon=":eye:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(PAGE_STYLE, unsafe_allow_html=True)
# ========================#

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
FOLDER_PATH = "."

# Initialize streamlit session_state
if "keywordSearch" not in st.session_state:
    st.session_state["keywordSearch"] = "na"


def main():
    # Main panel
    st.title(":eye: HindsightAI")
    st.write("### Searching Videos with Machine Learning")
    video_element = st.empty()
    keyword_search = st.empty()

    # Side panel
    st.sidebar.title("Load Media:")
    valid_files = videoCompatabilityCheck(os.listdir(FOLDER_PATH))
    valid_files.insert(0, "Select a file...")
    selected_filename = st.sidebar.selectbox("Select a video to search...", valid_files)
    granularity = st.sidebar.selectbox(
        "Select the granularity of search...", ["Low", "Medium", "High"]
    )

    if granularity == "Low":
        num_segments = 4
    elif granularity == "Medium":
        num_segments = 9
    elif granularity == "High":
        num_segments = 16

    # Reactivity
    if selected_filename == "Select a file...":
        st.write("## Please select a valid video file to analyze")

    if selected_filename != "Select a file...":
        video_path = os.path.join(FOLDER_PATH, selected_filename)
        video_object = video_element.video(loadVideo(video_path))
        framerate = getFramerate(video_path)

        selected_framerate = st.sidebar.slider(
            "Adjust frames sampled per second (fps)",
            min_value=1,
            max_value=framerate,
            value=1,
            step=1,
        )
        prediction_confidence = st.sidebar.slider(
            "Adjust predicton confidence (%) threshold",
            min_value=70,
            max_value=95,
            value=80,
            step=1,
        )

        tensorDict = videoToTensor(video_path, framerate, selected_framerate)
        st.success(f"Completed processing {len(list(tensorDict.items()))} frames!")
        keyword_search = st.text_input(label="Search video by keyword...")

        if keyword_search:
            selected_result = st.empty()

            # Run clip if a clip result is not cached or if query changes.
            if (
                st.session_state["keywordSearch"] == "na"
                or st.session_state["keywordSearch"] != keyword_search
            ):
                st.session_state["keywordSearch"] = keyword_search
                search_phrase = st.empty()
                text = [keyword_search, "."]

                with search_phrase:
                    st.write(f"Searching the video for {keyword_search}...")

                unfiltered = clipAnalyze(tensorDict, text, num_segments=num_segments)
                clipResult = filterResult(unfiltered, prediction_confidence / 100)

                st.session_state["result"] = clipResult.copy()
                st.session_state["unfiltered"] = unfiltered.copy()
                search_phrase.empty()
                selected_result = st.selectbox("Analysis results:", clipResult.keys())

            # Use cached result if one exists and the query has not changed.
            else:
                clipResult = st.session_state["result"]
                unfiltered = st.session_state["unfiltered"]
                selected_result = st.selectbox("Analysis results:", clipResult.keys())

            # Display where selected result occurs in the video.
            if selected_result:
                st.write(f"{keyword_search} found in frame", selected_result)

                video_element.empty()
                video_object = video_element.video(
                    loadVideo(video_path),
                    start_time=int(int(selected_result) // framerate),
                )

                y1 = unfiltered[selected_result][1][0]
                y2 = unfiltered[selected_result][1][1]
                x1 = unfiltered[selected_result][1][2]
                x2 = unfiltered[selected_result][1][3]

                frametoDisplay = Image.fromarray(
                    cv2.rectangle(
                        tensorDict[selected_result],
                        (x1, y1),
                        (x2, y2),
                        (255, 40, 50),
                        2,
                    ),
                    mode="RGB",
                )
                st.image(frametoDisplay, channels="RGB")

                fig = px.bar(
                    x=list(unfiltered.keys()),
                    y=[i[0] for i in unfiltered.values()],
                    labels={"x": "Frame", "y": "Confidence"},
                )
                st.write(fig)

    return None


def videoCompatabilityCheck(filename_list):
    """

    Returns all '.mp4' extension files from current directory.
    Does not account for encoding, all encodings are welcome here.
    
    """

    valid_files = [file for file in filename_list if file.endswith(".mp4")]

    if len(valid_files) == 0:
        valid_files = ["No suitable files..."]

    return valid_files


def loadVideo(video_path: str):
    """
    """

    video_bytes = open(video_path, "rb").read()

    return video_bytes


def getFramerate(videoPath: str):
    """
    """

    vr = VideoReader(videoPath, ctx=cpu(0))
    framerate = int(vr.get_avg_fps())

    return framerate


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def videoToTensor(videoPath: str, framerate: int, selected_fps: int = 1):
    """
    Takes in a valid .mp4 path and converts it 
    to a dict of frame tensors using the glorious
    decord package - blessed be.

    Arguments:
    ==========
        videoPath: str
            a relative path to an .mp4 video file.

        fps: int
            Not implemented yet. Will use to
            implement user selectable fps.
    

    Returns:
    ========
        tensorDict: dict
            A dict of video frame tensors e.g.:
                tensordict = {
                    'timeInseconds' = np.tensor(frameTensor)
                }
        
        framerate: int
            an integer representing the average frame rate
            of the decode video object.
    
    """

    tensorDict = {}
    vr = VideoReader(videoPath, ctx=cpu(0))
    framehash = np.arange(1, framerate + 1)[::-1]
    selected_framerate = framehash[selected_fps - 1]

    for framecount in range(len(vr)):
        if framecount % selected_framerate == 0:
            tensorDict[str(framecount)] = vr[framecount].asnumpy()

    return tensorDict


def clipAnalyze(tensorDict, searchString: str, num_segments: int):
    """

    Runs a clip analysis on each frame of the provided
    dictionary's values where the posisble text classes 
    are the user provided string. Also, renders a streamlit 
    progress bar that scales to the length of the analysis.

    Arguments:
    ==========
        tensorDict: dict
            A dictionary of video frame tensors e.g.:
                tensordict = {
                    'timeInseconds' = np.tensor(frameTensor)
                }

        searchString: str
            A user supplied string to tokenize and encode
            for use as a clip imbedding for analysis.

    Returns:
    ========
        sorted_result: dict
            a dictonary of frame indicies and their corresponding
            clip probabilities in descending order of probability e.g.:
                sorted_result = {
                    'timeInSeconds' = float32(probability)
                }
    """

    result = {}

    progress_bar = st.empty()
    progress_bar = st.progress(0)
    progress_bar_value = 0
    # scale progressbar increment value to the len(video)
    incrementValue = 100 / len(list(tensorDict.items())) / 100

    tokenizedText = clip.tokenize(searchString).to(DEVICE)

    with torch.no_grad():
        for frameTensorPair in tensorDict.items():
            progress_bar.progress(progress_bar_value)
            im = frameTensorPair[1]
            segments = []
            indices = []

            for i in segment_np_array(im, num_segments):
                segments.append(im[i[0] : i[1], i[2] : i[3], :])
                indices.append([i[0], i[1], i[2], i[3]])

            probs = []

            for segment in segments:
                image = PREPROCESS(Image.fromarray(segment)).unsqueeze(0).to(DEVICE)
                logits_per_image, __ = MODEL(image, tokenizedText)
                prob = logits_per_image.softmax(dim=-1).cpu().numpy()
                probs.append(np.round(prob[0][0], 2))

            prob = max(probs)
            max_index = probs.index(prob)
            index = indices[max_index]

            result[str(frameTensorPair[0])] = [prob, index]

            progress_bar_value += incrementValue

    progress_bar.empty()

    return result


def filterResult(result, confidenceLevel: float = 0.90):
    """
    Arguments:
    ==========
        sortedResult: dict
            A dictionary of frame-probability pairs e.g.:
                sorted_result = {
                    'timeInseconds' = np.float(probability)
                }

        confidenceLevel: float
            A user supplied float that acts as a low end
            cut off with which to filter results.

    Returns:
    ========
        filteredResult: dict
            A dictionary of frame-probability pairs e.g.:
                sorted_result = {
                    'timeInseconds' = np.float(probability)
                }
    
    """
    sorted_result = dict(sorted(result.items(), key=lambda x: x[1][0], reverse=True))

    filteredResult = {}

    for frameTensorPair in sorted_result.items():
        if frameTensorPair[1][0] > confidenceLevel:
            filteredResult[frameTensorPair[0]] = frameTensorPair[1]

        elif len(filteredResult) == 0:
            return (
                filteredResult["No results with the selected confidence level"] == None
            )

        else:
            return filteredResult


def segment_np_array(arr, num_segments):
    """
    Takes in a numpy array for a 3-channel image and a desired number
    of segments for the image. 
    
    Returns a list of tuples denoting the (top, bottom, left, right) indexes
    for each image segment.
    
    *** Must take in only perfect squares as the "num_segments"!***
    
    """

    box_dim = int(np.sqrt(num_segments))

    image_h = arr.shape[0]
    image_w = arr.shape[1]
    kernel_size = int(np.floor(image_w / (box_dim * 0.83)))

    if kernel_size > image_h:
        kernel_size = image_h

    stride_w = int(np.floor((image_w - kernel_size) / (box_dim - 1)))
    stride_h = int(np.floor((image_h - kernel_size) / (box_dim - 1)))

    segments = []
    top = 0

    for i in range(box_dim):
        bottom = top + kernel_size
        left = 0
        for j in range(box_dim):
            right = left + kernel_size
            segments.append((top, bottom, left, right))
            left += stride_w

        top += stride_h

    return segments


if __name__ == "__main__":
    main()
