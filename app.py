import streamlit as st
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import os

#========================#
st.set_page_config(
    page_title="HindsightAI",
    page_icon=":eye:",
    layout="wide",
    initial_sidebar_state="expanded")
    
PAGE_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(PAGE_STYLE, unsafe_allow_html=True)
#========================#  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
FOLDER_PATH = '.'

def main():
    # Main panel
    st.title(":eye: HindsightAI")
    st.write("### Searching Videos with Machine Learning")
    video_placeholder= st.empty()
    keyword_search = st.empty()

    # Side panel
    st.sidebar.title("Load Media:")
    valid_files = videoCompatabilityCheck(os.listdir(FOLDER_PATH))
    valid_files.insert(0, 'Select a file...')
    selected_filename = st.sidebar.selectbox('Select a video to analyze...', valid_files)

    if selected_filename == 'Select a file...':
        st.write('## Please select a valid video file to analyze')
    else:
        st.write('')

    if selected_filename != 'Select a file...':
        video_path = os.path.join(FOLDER_PATH, selected_filename)
        video_placeholder = st.video(loadVideo(video_path))
        
    
        tensorDict = videoToTensor(video_path)    
        st.success(f'Completed processing {len(list(tensorDict.items()))} frames!')


        # Appears after video is initially processed
        keyword_search = st.text_input(label="Search video by keyword...")
        selected_framerate = st.sidebar.slider('Adjust frames sampled per second (FPS)',min_value=1, max_value=30, value=1, step=1) # work in progress
        prediction_confidence = st.sidebar.slider('Adjust predicton confidence (%)',min_value=70, max_value=95, value=80, step=1)

        
        if keyword_search:
            search_phrase = st.empty()
            selected_result = st.empty()
            text = [keyword_search,'.']

            with search_phrase:
                st.write(f'Searching the video for {keyword_search}...')


            clipResult = filterResult(clipAnalyze(tensorDict,text), prediction_confidence/100)
            
            search_phrase.empty()
            selected_result = st.selectbox('', clipResult.keys())

            if selected_result:
                st.write(f'{keyword_search} found in frame', selected_result)
                
                # video result
                #video_placeholder.empty()
                #video_placeholder = st.video(loadVideo(video_path), start_time= int(selected_result)//30)
                
                frametoDisplay = Image.fromarray(tensorDict[selected_result],mode='RGB')
                st.image(frametoDisplay, channels="RGB")

    return None


def videoCompatabilityCheck(filename_list):
    '''

    Returns all '.mp4' extension files from current directory.
    Does not account for encoding, all encodings are welcome here.
    
    '''
      
    valid_files = [file for file in filename_list if file.endswith('.mp4')]
    
    if len(valid_files) == 0:
        valid_files = ['No suitable files...']
    
    return valid_files


def loadVideo(video_path:str):
    '''
    '''
    video_bytes = open(video_path, 'rb').read()
    return video_bytes


@st.cache(suppress_st_warning=True)
def videoToTensor(videoPath:str, fps:int = 1):
    '''
    Takes in a valid .mp4 path and converts it 
    to a dict of frame tensors using the glorious
    decord package blessed be.

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
    
    '''
    tensorDict = {}
    vr = VideoReader(videoPath, ctx=cpu(0))
    cvr = cv2.VideoCapture(videoPath)

    framespersecond= int(cvr.get(cv2.CAP_PROP_FPS))

    print("The total number of frames in this video is ", fps)


    for i in range(len(vr)):
        if i % 30 == 0:
            tensorDict[str(i)] = vr[i].asnumpy()
    
    return tensorDict


def clipAnalyze(tensorDict, searchString:str):
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

    Returns:
    ========
        sorted_result: dict
            a dictonary of frame indicies and their corresponding
            clip probabilities in descending order of probability e.g.:
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

    tokenizedText = clip.tokenize(searchString).to(DEVICE)   

    with torch.no_grad():
        for frameTensorPair in tensorDict.items():
            progress_bar.progress(progress_bar_value)
            
            image = PREPROCESS(Image.fromarray(frameTensorPair[1])).unsqueeze(0).to(DEVICE)
            logits_per_image, __ = MODEL(image, tokenizedText)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            result[str(frameTensorPair[0])] = np.round(probs[0][0],2)
            
            progress_bar_value += incrementValue
    
    sorted_result = dict(sorted(result.items(), key=lambda x: x[1],reverse=True))
    progress_bar.empty()

    return sorted_result


def filterResult(sorted_result, confidenceLevel:float = 0.90):
    '''
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
    
    '''
    filteredResult = {}

    for frameTensorPair in sorted_result.items():
        if frameTensorPair[1] > confidenceLevel:
            filteredResult[frameTensorPair[0]] = frameTensorPair[1]
        
        elif len(filteredResult) == 0: 
            return filteredResult['No results with the selected confidence level'] == None
        
        else:
            return filteredResult


if __name__ == '__main__':
    main()
