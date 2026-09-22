from dotenv import load_dotenv
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=250
)

model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()
text="Support Vector Machine (SVM) is a supervised machine learning algorithm used mainly for classification, although it can also be used for regression. The main idea of SVM is to find the optimal hyperplane that separates data points belonging to different classes. Among multiple possible separating hyperplanes, SVM chooses the one that maximizes the margin, which is the distance between the hyperplane and the closest data points from each class. These closest data points are called support vectors. For data that is not linearly separable, SVM can use the kernel trick to transform the data into a higher-dimensional space where a separating boundary may be found. Common kernels include linear, polynomial, and radial basis function (RBF) kernels. The parameter C controls the trade-off between maximizing the margin and correctly classifying training samples, while gamma controls the influence of individual training points when using kernels s" \
"uch as RBF. SVM is particularly effective for high-dimensional datasets and can work well when there is a clear separation between classes."
prompt1=PromptTemplate(
    template="Generate quick notes from text \n {text}",
    input_variables=['text']
)
prompt2=PromptTemplate(
    template="Generate 5 question and answers from the given text\n  {text}",
    input_variables=['text']
)
prompt3=PromptTemplate(
    template="merge the provided notes and quiz questions into single document \n notes->{notes} , quiz=>{quiz}",
    input_variables=['notes','quiz']
)
# parallel chain
parallel_chain= RunnableParallel({
    'notes':prompt1 | model | parser,
    'quiz':prompt2  | model | parser
})
# single doc chain
merge_chain= prompt3 | model | parser
chain= parallel_chain | merge_chain
res=chain.invoke({'text':text})
print(res)


chain.get_graph().print_ascii();
