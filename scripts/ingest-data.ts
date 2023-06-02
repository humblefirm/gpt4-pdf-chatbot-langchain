import natural from 'natural';
import * as stopword from 'stopword';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { pinecone } from '@/utils/pinecone-client';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';

const filePath = 'docs';

const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;

export const run = async () => {
  try {
    const directoryLoader = new DirectoryLoader(filePath, {
      '.pdf': (path) => new PDFLoader(path),
    });

    const rawDocs = await directoryLoader.load();

    // Normalize, remove stopwords, and stem the text
    const processedDocs = rawDocs.map((doc) => {
      let text = String(doc).toLowerCase();
      let tokens = tokenizer.tokenize(text);
      tokens = stopword.removeStopwords(tokens as string[]);
      tokens = tokens.map((token) => stemmer.stem(token));
      return { ...doc, text: tokens.join(' ') };
    });

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 400,
    });

    const docs = await textSplitter.splitDocuments(processedDocs);
    console.log('split docs', docs);

    console.log('creating vector store...');
    const embeddings = new OpenAIEmbeddings();
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    await PineconeStore.fromDocuments(docs, embeddings, {
      pineconeIndex: index,
      namespace: PINECONE_NAME_SPACE,
      textKey: 'text',
    });
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
  console.log('ingestion complete');
})();
