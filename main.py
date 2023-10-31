pip install -qU transformers==4.33.0 torch==2.0.1 accelerate pyspark einops langchain xformers bitsandbytes faiss-gpu  pypdf2 huggingface tokenizers
dbutils.library.restartPython()
from torch import cuda, bfloat16
import transformers
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    'meta-llama/Llama-2-13b-chat-hf',
    use_auth_token=hf_auth
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-13b-chat-hf',
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-13b-chat-hf',
    use_auth_token=hf_auth
)
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids
import torch
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids
from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])
generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    stopping_criteria=stopping_criteria,  # without this model rambles during chat(like one shared by debarshi)
    temperature=0.2,  # 'randomness' of outputs, 0.01 min
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from PyPDF2 import PdfReader
def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase
pdf_reader = PdfReader('/dbfs/FileStore/tables/Infosys_Earnings_Call_Q4_FY22.pdf')
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()
knowledgeBase = process_text(text)
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generate_text)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
template = """Use the following pieces of context to answer the question at the end. If the answer can't be determined using only the information in the provided context simply output "The given document does not have this information", do not try to make up an answer.
{context}
Question: {question} 
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

chain= RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=knowledgeBase.as_retriever(),
    chain_type_kwargs={
        "prompt": QA_CHAIN_PROMPT
    }
)
from langchain.callbacks import get_openai_callback
def resp(query,chat_history):        
        if query:
            with get_openai_callback() as cost:
                response = chain.run(query=query,chat_history=chat_history)                
            return(response)
import sys
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result=resp(query,chat_history)
    print('Answer: ' + result+'\n')
    print('Most relevant document: ' + str(knowledgeBase.similarity_search_with_relevance_scores(query)[0][0])[14:-1]+'\n')
    print('Score: ' + str(knowledgeBase.similarity_search_with_relevance_scores(query)[0][-1])+'\n')
    chat_history.append((query, result))
