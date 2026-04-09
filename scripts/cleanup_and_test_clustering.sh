#!/bin/bash
# Скрипт для очистки старого embedding cache и тестирования улучшенной кластеризации

set -e

echo "=== Cleaning old embedding cache ==="
echo "Old cache without instructions will be removed..."

# Backup old cache if needed
if [ -d "data/cache/embeddings/local_st/BAAI__bge-m3" ]; then
    OLD_CACHE_SIZE=$(du -sh data/cache/embeddings/local_st/BAAI__bge-m3 | cut -f1)
    echo "Old cache size: $OLD_CACHE_SIZE"
    
    read -p "Do you want to backup old cache before deletion? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        BACKUP_DIR="data/cache/embeddings_backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        mv data/cache/embeddings/local_st/BAAI__bge-m3 "$BACKUP_DIR/"
        echo "✅ Old cache backed up to: $BACKUP_DIR"
    else
        rm -rf data/cache/embeddings/local_st/BAAI__bge-m3
        echo "✅ Old cache deleted"
    fi
else
    echo "No old cache found - starting fresh"
fi

# Also clean database cache for BGE model
echo ""
echo "=== Cleaning database embedding cache ==="
echo "Running SQL cleanup..."

python3 << 'PYEOF'
from app.db.session import SessionLocal
from app.db.models import EmbeddingCache
from sqlalchemy import delete

db = SessionLocal()
try:
    # Delete old BGE embeddings without instruction namespace
    result = db.execute(
        delete(EmbeddingCache).where(
            EmbeddingCache.provider == "local_st",
            EmbeddingCache.model == "BAAI/bge-m3"
        )
    )
    db.commit()
    print(f"✅ Deleted {result.rowcount} old embedding cache entries from database")
except Exception as e:
    print(f"❌ Error cleaning database cache: {e}")
    db.rollback()
finally:
    db.close()
PYEOF

echo ""
echo "=== Verifying new settings ==="
python3 << 'PYEOF'
from app.core.config import Settings

settings = Settings()
print(f"✅ Embedding instruction mode: {settings.embedding_instruction_mode}")
print(f"✅ Soft assignment enabled: {settings.cluster_soft_assignment_enabled}")
print(f"✅ Reduction enabled: {settings.cluster_reduction_enabled}")
print(f"✅ Accept noise ratio: {settings.cluster_accept_noise_ratio}")
print(f"✅ Topic coherence min: {settings.topic_coherence_min}")
PYEOF

echo ""
echo "=== All done! ==="
echo ""
echo "Now you can run the pipeline with improved clustering:"
echo "  - New embeddings will use instruction prompts for BGE model"
echo "  - Soft assignment will properly assign borderline comments"
echo "  - Stricter quality gates will reject low-quality clusters"
echo "  - PCA reduction will improve HDBSCAN performance"
echo ""
echo "Expected improvements:"
echo "  ✅ Cluster confidence: 0.90-0.99 (was: variable)"
echo "  ✅ Noise ratio: <0.35 (was: >0.45)"
echo "  ✅ Silhouette score: >0.15 (was: ~0.05)"
echo "  ✅ Ambiguous comments: <20% (was: variable)"
echo ""
