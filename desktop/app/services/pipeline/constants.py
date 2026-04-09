"""Module-level constants for the analysis pipeline.

Centralises compiled regex patterns used for comment classification
(author references, agreement/disagreement detection, offensive language)
and numeric thresholds that govern cluster pruning behaviour.

String constants such as ``_UNDETERMINED_POSITION_KEY`` and
``_CLUSTER_STOPWORDS`` live in ``text_utils.py`` instead.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Regex patterns for comment classification
# ---------------------------------------------------------------------------

# Matches second-person pronoun references to the video author.
# Author name matching is done dynamically in ReportBuilder.has_author_reference().
_AUTHOR_REFERENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(вы|вас|вам|ваш)\b",
    re.IGNORECASE,
)

# Detects expressions of disagreement or criticism directed at the author.
_DISAGREEMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"не\s+соглас\w*|неправ|ошиба|вводите\s+в\s+заблуждени|ложь|вр[её]т|манипул|"
    r"бред|чушь|пропаганд|предвзят|не\s+верю|искажа|перекручива"
    r")\b",
    re.IGNORECASE,
)

# Detects expressions of agreement or support for the author's position.
_AGREEMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"согласен|поддерживаю|вы\s+правы|правы|верно|точно\s+сказано|хороший\s+разбор|"
    r"thank you|good point|agreed"
    r")\b",
    re.IGNORECASE,
)

# Detects offensive language combined with disagreement (Russian and English).
_OFFENSIVE_DISAGREEMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"\u0431\u043b\u044f\u0434|\u0431\u043b\u044f|\u0441\u0443\u043a\u0430|\u0441\u0443\u0447\u043a|"
    r"\u0445\u0443\u0439|\u0445\u0443\u0435|\u043f\u0438\u0437\u0434|\u0435\u0431\u0430\u043d|\u0435\u0431\u0430\u0442|"
    r"\u043c\u0443\u0434\u0430\u043a|\u0434\u0435\u0431\u0438\u043b|\u0438\u0434\u0438\u043e\u0442|\u0442\u0432\u0430\u0440\u044c|\u0443\u0431\u043b\u044e\u0434"
    r"|fuck|fucking|shit|bitch|asshole|moron|idiot"
    r")\b",
    re.IGNORECASE,
)

# Extracts word tokens (4+ characters) for cluster keyword extraction.
_CLUSTER_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w{4,}", re.UNICODE)

# Strips non-word characters for text matching / deduplication.
_MATCH_TEXT_RE: Final[re.Pattern[str]] = re.compile(r"[\W_]+", re.UNICODE)

# Extracts all word tokens from a generated cluster title for validation.
_TITLE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)

# Tokens that indicate a low-quality or overly generic cluster title.
_TITLE_BAD_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"разное|прочее|misc|other|комментарии|обсуждение|разбор|новости|видео|политика|"
    r"спасибо|дело|меня|ваши|есть|просто|вообще"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Cluster pruning thresholds
# ---------------------------------------------------------------------------

# Minimum number of comments a cluster must contain to be eligible for pruning
# of its weakest members. Smaller clusters are left untouched.
_CLUSTER_PRUNE_MIN_SIZE: Final[int] = 8

# Quantile of intra-cluster cosine similarities below which a comment is
# considered a weak member (candidate for removal).
_CLUSTER_PRUNE_SIMILARITY_QUANTILE: Final[float] = 0.25

# Absolute floor for cosine similarity: comments below this threshold are
# pruned regardless of the quantile cutoff.
_CLUSTER_PRUNE_MIN_SIMILARITY: Final[float] = 0.18

# Maximum fraction of a cluster's comments that may be dropped in a single
# pruning pass, to avoid over-aggressive removal.
_CLUSTER_PRUNE_MAX_DROP_SHARE: Final[float] = 0.35
