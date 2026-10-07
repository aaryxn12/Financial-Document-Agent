FROM public.ecr.aws/lambda/python:3.13

# Chroma caches its embedding model under $HOME/.cache. Lambda's filesystem is
# read-only outside /tmp, so we fetch the model at build time into a fixed path.
ENV HOME=/opt/home
RUN mkdir -p $HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model once, at build time.
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warm up'])" \
    && chmod -R a+rX $HOME
COPY . .

CMD ["api.main.handler"]