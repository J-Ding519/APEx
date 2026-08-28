#!/bin/bash
# Search-R1 本地检索服务 :8003

BASE_DIR="/path/to/workdir"
SEARCH_R1_DIR="${BASE_DIR}/Search-R1"
OTHER_DIR="${BASE_DIR}/MIA-other"

E5_MODEL="${OTHER_DIR}/e5-base-v2"
FAISS_INDEX="${OTHER_DIR}/faiss_index/e5_Flat.index"
WIKI25_MERGED="${OTHER_DIR}/wiki25_merged/wiki25.jsonl"

# 合并 wiki25（首次运行）
if [ ! -f "${WIKI25_MERGED}" ]; then
    echo "合并 wiki25 分片文件..."
    mkdir -p "$(dirname ${WIKI25_MERGED})"
    cat ${OTHER_DIR}/wiki25/wiki25_part_* > "${WIKI25_MERGED}"
    echo "合并完成: $(du -sh ${WIKI25_MERGED} | cut -f1)"
fi

# 检查 FAISS 索引（首次运行需构建）
if [ ! -f "${FAISS_INDEX}" ]; then
    echo "构建 FAISS 索引..."
    mkdir -p "$(dirname ${FAISS_INDEX})"
    cd "${SEARCH_R1_DIR}"
    python search_r1/search/index_builder.py \
        --retrieval_method e5 \
        --model_path "${E5_MODEL}" \
        --corpus_path "${WIKI25_MERGED}" \
        --save_dir "$(dirname ${FAISS_INDEX})" \
        --use_fp16 \
        --max_length 256 \
        --batch_size 512 \
        --pooling_method mean \
        --faiss_type Flat \
        --save_embedding
    echo "FAISS 索引构建完成"
fi

cd "${SEARCH_R1_DIR}"
python search_r1/search/retrieval_server.py \
    --index_path "${FAISS_INDEX}" \
    --corpus_path "${WIKI25_MERGED}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}"
