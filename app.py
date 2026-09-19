import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# A small instruction-tuned chat model suitable for a classroom/local demo.
# The first run downloads the model from Hugging Face and caches it locally.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


st.set_page_config(
    page_title="Hugging Face Chatbot",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading the Hugging Face model...")
def load_model():
    """Load the tokenizer and model once and reuse them across Streamlit reruns."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # "auto" uses an appropriate numerical precision for the computer.
        torch_dtype="auto",
        # "auto" uses a GPU when available and otherwise runs on the CPU.
        device_map="auto",
    )

    return tokenizer, model


def generate_reply(tokenizer, model, messages, max_new_tokens, temperature):
    """Convert the conversation to model input and generate one assistant reply."""

    # Keep the most recent messages so the prompt does not grow without limit.
    recent_messages = messages[-10:]

    conversation = [
        {
            "role": "system",
            "content": (
                "You are a helpful teaching assistant. Give clear, concise and "
                "accurate answers. If you are uncertain, say so."
            ),
        }
    ] + recent_messages

    # The chat template converts role-based messages into the exact text format
    # expected by this particular instruction-tuned model.
    model_inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,  # Adds the marker that begins the assistant reply.
        return_tensors="pt",         # Returns PyTorch tensors.
        return_dict=True,
    )

    # Move the input tensors to the same device as the model (CPU or GPU).
    model_inputs = model_inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,

            # Maximum number of new tokens in the assistant's answer.
            # It does not include the existing conversation tokens.
            max_new_tokens=max_new_tokens,

            # Sampling allows varied answers instead of always selecting the
            # single most likely next token.
            do_sample=True,

            # Lower values are more focused; higher values are more creative.
            temperature=temperature,

            # Select the next token from a probability group covering 90% of
            # the model's likely choices.
            top_p=0.90,

            # Slightly discourages repetitive wording.
            repetition_penalty=1.10,

            # Tells generation which token can be used for sequence padding.
            pad_token_id=tokenizer.eos_token_id,
        )

    # The generated tensor contains both the original prompt and the new answer.
    # Remove the prompt tokens and keep only the assistant's newly generated IDs.
    input_length = model_inputs["input_ids"].shape[1]
    reply_ids = generated_ids[0][input_length:]

    # Convert the output token IDs back into readable text.
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
    return reply


st.title("🤖 Hugging Face Transformer Chatbot")
st.caption(f"Local Streamlit chatbot powered by `{MODEL_NAME}`")


with st.sidebar:
    st.header("Generation settings")

    max_new_tokens = st.slider(
        "Maximum new tokens",
        min_value=32,
        max_value=512,
        value=192,
        step=32,
        help="Controls the maximum length of each new assistant response.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower is more focused; higher is more varied.",
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.info(
        "The first launch downloads the model. CPU generation may take a few "
        "seconds; a supported GPU is faster."
    )


# Streamlit reruns the script after each interaction. Session state preserves
# the conversation between those reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display all previous user and assistant messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


tokenizer, model = load_model()


# st.chat_input stays fixed at the bottom of the app.
if prompt := st.chat_input("Type your message here..."):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = generate_reply(
                    tokenizer,
                    model,
                    st.session_state.messages,
                    max_new_tokens,
                    temperature,
                )
            except Exception as error:
                st.error(f"The model could not generate a response: {error}")
                st.stop()

        if not reply:
            reply = "I could not generate a response. Please try another prompt."

        st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
