import streamlit as st
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import os
import plotly.express as px

# ======================== #
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
# ======================== #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
FOLDER_PATH = "."

if "keywordSearch" not in st.session_state:
    st.session_state["keywordSearch"] = None

def main():
    # Main panel
    st.title(":eye: HindsightAI")
    subtitle = st.empty()
    video_element = st.empty()
    keyword_search = st.empty()
    splash_text = st.empty()
    subtitle.write("### Search Any Video for Anything")
    splash_text.write('## Please select a video to analyze!')

    # Side panel
    st.sidebar.title("Load Media:")
    valid_files = videoCompatabilityCheck(os.listdir(FOLDER_PATH))
    valid_files.insert(0, "Select a file...")
    selected_filename = st.sidebar.selectbox("Select a video to search:", valid_files)    

    num_segments = 9

    # Reactivity
    if selected_filename != "Select a file...":
        subtitle.empty()
        splash_text.empty()

        # Instantiate videoplayer
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
            max_value=100,
            value=85,
            step=1,
        )

        tensorDict = videoToTensor(video_path, framerate, selected_framerate)
        st.success(f"Completed processing {len(list(tensorDict.items()))} frames!")
        keyword_search = st.text_input(label="Search video by keyword...")

        if keyword_search:
            selected_result = st.empty()

            # Run clip if a clip result is not cached or if query changes.
            if (
                st.session_state["keywordSearch"] == None
                or st.session_state["keywordSearch"] != keyword_search
            ):
                st.session_state["keywordSearch"] = keyword_search
                search_phrase = st.empty()
                text = [keyword_search, "."]

                with search_phrase:
                    st.write(f"Searching the video for {keyword_search}...")

                unfiltered = clipDict(tensorDict, text, num_segments=num_segments)

                try:
                    clipResult = filterResult(unfiltered, prediction_confidence / 100)

                except KeyError:
                    st.error('No results found at the selected confidence level')
                    st.stop()

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
                st.write(
                    f"""{keyword_search} found in frame {selected_result},
                     with {int(clipResult[selected_result][0]*100)}% confidence"""
                )

                video_element.empty()
                video_object = video_element.video(
                    loadVideo(video_path),
                    start_time=int(int(selected_result) // framerate),
                )

                # Display frame with segmentation bounding box
                y1,y2,x1,x2 = unfiltered[selected_result][1]
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


def clipAnalyze(tensor, searchString:str):
    '''
    Arguments:
    ==========
        tensor: tensor
            A tensor from a single frame in the video 
            being analyzed.

        searchString: str
            A user supplied string to tokenize and encode
            for use as a clip imbedding for analysis.


    Returns:
    ========
        result: float
            A float representing the confidence calculated 
            by clip, given the user's text query.
    '''
    
    tokenizedText = clip.tokenize(searchString).to(DEVICE)  
    
    with torch.no_grad():
        image = PREPROCESS(Image.fromarray(tensor)).unsqueeze(0).to(DEVICE)
        logits_per_image, __ = MODEL(image, tokenizedText)
        prob = logits_per_image.softmax(dim=-1).cpu().numpy()
        result = np.round(prob[0][0],2)
        
    return result


def clipDict(tensorDict, searchString:str, num_segments:int, confidenceLevel:float = 0.90):
    '''

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

        num_segments: str
            Number of segments by which to split each 
            of the frames for more detailed analysis.

    Returns:
    ========
        result: dict
            a dictonary of frame indicies and their corresponding
            clip probabilities e.g.:
                sorted_result = {
                    'timeInSeconds' = float32(probability)
                }
    '''

    result = {}

    progress_bar = st.empty()
    progress_bar = st.progress(0)
    progress_bar_value = 0
    # scale progressbar increment value to the len(video)
    incrementValue = (100/len(list(tensorDict.items()))/100) 


    for frameTensorPair in tensorDict.items():
        progress_bar.progress(progress_bar_value)
        
        # Analyze entire image with CLIP
        im = frameTensorPair[1]
        prob = clipAnalyze(im, searchString)
        result[str(frameTensorPair[0])] = [prob, [0, im.shape[0], 0, im.shape[1]]]

            
        # If image is a match to text, analyze segments of image
        if result[str(frameTensorPair[0])][0] >= confidenceLevel:
            highest_conf = result[str(frameTensorPair[0])][0]

            x1, x2, y1, y2 = None, None, None, None

            while True:

                segments = []
                indices = []
                for i in segment_np_array(im, num_segments, y1, y2, x1, x2):
                    segments.append(im[i[0]:i[1],i[2]:i[3],:])
                    indices.append([i[0],i[1],i[2],i[3]])

                probs = []

                for segment in segments:
                    probs.append(clipAnalyze(segment, searchString))

                prob = max(probs)
                max_index = probs.index(prob)
                index = indices[max_index]

                if prob < highest_conf - 0.05:
            	    break

                result[str(frameTensorPair[0])] = [prob, index]

                y1 = index[0]
                y2 = index[1]
                x1 = index[2]
                x2 = index[3]
            
            progress_bar_value += incrementValue
    
    progress_bar.empty()

    return result


def filterResult(result, confidenceLevel:float = 0.90):
    '''
    Arguments:
    ==========
        result: dict
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
    
    '''
    sorted_result = dict(sorted(result.items(), key=lambda x: x[1][0],reverse=True))

    filteredResult = {}

    for frameTensorPair in sorted_result.items():
        if frameTensorPair[1][0] > confidenceLevel:
            filteredResult[frameTensorPair[0]] = frameTensorPair[1]
        
        elif len(filteredResult) == 0: 
            return filteredResult['No results with the selected confidence level'] == None
        
        else:
            return filteredResult


def segment_np_array(arr, num_segments, y1=None, y2=None, x1=None, x2=None):
    
    '''
    Takes in a numpy array for a 3-channel image and a desired number
    of segments for the image. 
    
    Returns a list of tuples denoting the (top, bottom, left, right) indexes
    for each image segment.
    
    *** Must take in only perfect squares as the "num_segments"!***
    
    '''
    
    if x1 == None:
        x1 = 0
        y1 = 0
        x2 = arr.shape[1]
        y2 = arr.shape[0]
        
    box_dim = int(np.sqrt(num_segments))
    
    image_h = y2 - y1
    image_w = x2 - x1
    kernel_size = int(np.floor(image_w/(box_dim*0.80)))
    
    if kernel_size > image_h:
        kernel_size = image_h
    
    stride_w = int(np.floor((image_w - kernel_size)/(box_dim-1)))
    stride_h = int(np.floor((image_h - kernel_size)/(box_dim-1)))
    
    segments = []
    top = y1
    
    for i in range(box_dim):
        bottom = top + kernel_size
        left = x1
        for j in range(box_dim):
            right = left + kernel_size
            segments.append((top,bottom,left,right))
            left += stride_w
            
        top += stride_h
            
    return segments


if __name__ == "__main__":
    main()
