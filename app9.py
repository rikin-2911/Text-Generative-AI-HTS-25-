import streamlit as st
import torch
from torchtext.vocab import Vocab
from model_definition import TextGenerationLSTM
from PIL import Image

#Page
st.set_page_config(
    page_title="Generative AI Text Generation",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logo
st.sidebar.image("ai_logo.webp", width=150)

# Sidebar
st.sidebar.title("🚀 Text Generation AI")

#Documentation Link
st.sidebar.markdown("### 📄 Documentation")
st.sidebar.markdown("[View .ipynb Documentation](https://github.com/rikin-2911/Text-Generative-AI-HTS-25-/blob/main/documentation.ipynb)", unsafe_allow_html=True)

# Model Selection
model_choice = st.sidebar.radio(
    "Choose a Model",
    ("Q&A", "Storytelling", "Dialogue"),
    index=0,
    help="Select the type of text you want to generate."
)

#User Input
st.title("Generative AI Text Generation")
st.markdown("""
    <style>
        .big-font {
            font-size:20px !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Enter a prompt to generate text:</p>', unsafe_allow_html=True)

input_text = st.text_area(
    "Prompt:",
    placeholder="Once upon a time...",
    max_chars=200,
    help="Enter the starting text for generating content. Maximum 200 characters."
)
generate_button = st.button("✨ Generate Text")

# Load Model and Vocabulary
@st.cache_resource
def load_model_and_vocab(model_choice):
    if model_choice == "Q&A":
        model = torch.load('models/genai_Q_A.pth', map_location=torch.device('cpu'), weights_only=False)
        vocab = torch.load('models/vocab_qa.pth', map_location=torch.device('cpu'))
    elif model_choice == "Storytelling":
        model = torch.load('models/genai_stories.pth', map_location=torch.device('cpu'), weights_only=False)
        vocab = torch.load('models/vocab_story.pth', map_location=torch.device('cpu'))
    elif model_choice == "Dialogue":
        model = torch.load('models/genai_englang.pth', map_location=torch.device('cpu'), weights_only=False)
        vocab = torch.load('models/vocab_dialogue.pth', map_location=torch.device('cpu'))

    model.eval()  
    return model, vocab

model, vocab = load_model_and_vocab(model_choice)

# Sidebar
length_options = {
    "Q&A": {"Short (20 words)": 20, "Medium (50 words)": 50, "Long (100 words)": 100},
    "Storytelling": {"Short (50 words)": 50, "Medium (100 words)": 100, "Long (200 words)": 200},
    "Dialogue": {"Short (10 words)": 10, "Medium (20 words)": 20, "Long (50 words)": 50}
}

length_choice = st.sidebar.selectbox(
    "Select Text Length",
    list(length_options[model_choice].keys()),
    index=1,
    help="Choose the length of the generated text."
)

# Get the selected number of words
num_words = length_options[model_choice][length_choice]

# Updated Text Generation Function
def generate_text(model, vocab, input_text, num_words):
    words = input_text.split()
    state_h, state_c = model.init_hidden(1)

    for _ in range(num_words):
        input_seq = torch.tensor([[vocab[word] for word in words]], dtype=torch.long)
        out, (state_h, state_c) = model(input_seq, (state_h, state_c))
        
        last_word_logits = out[0, -1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().tolist()
        word_idx = torch.multinomial(torch.tensor(p), 1).item()

        words.append(vocab.lookup_token(word_idx))

    return ' '.join(words)

# Generate Text and Display Output
if generate_button:
    if input_text.strip():
        with st.spinner("Generating text..."):
            generated_text = generate_text(model, vocab, input_text, num_words)
        st.subheader("✨ Generated Text:")
        st.success(generated_text)
        
        # Copy to Clipboard Button
        st.button("📋 Copy to Clipboard", on_click=lambda: st.code(generated_text, language='text'))
    else:
        st.warning("🚨 Please enter a prompt to generate text.")

# Team Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Team Name: MechaMinds")
st.sidebar.markdown("### 🤝 Team Members:")
team_members = {
    "RIKIN PITHADIA (Leader)": "https://www.linkedin.com/in/rikin-pithadia-20b94729b/",
    "DARJI KUNJ": "https://www.linkedin.com/in/kunj-darji-064b0a290",
    "PRAJAPATI PARTH": "https://www.linkedin.com/in/prajapati-parth-nareshbhai-07aa76340?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app",
    "YAGNIT BARAIYA": "https://www.linkedin.com/in/yagnit-baraiya-73421534a"
}

for name, link in team_members.items():
    st.sidebar.markdown(f"- [{name}]({link})")

# Feedback Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Feedback")
feedback_text = st.sidebar.text_area(
    "We value your feedback!",
    placeholder="Let us know your thoughts..."
)
feedback_button = st.sidebar.button("Submit Feedback")

if feedback_button:
    if feedback_text.strip():
        st.sidebar.success("✅ Thank you for your feedback!")
    else:
        st.sidebar.warning("🚨 Please enter your feedback before submitting.")