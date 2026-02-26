"""
AI Code Review — called by GitHub Actions CI.
Usage: python ai_review.py <review_type>
review_type: security | performance | best-practices

Exit 0 = no critical issues.
Exit 1 = critical issues found (fails the CI job).
"""
import json
import os
import subprocess
import sys

import boto3

# ─── Config ────────────────────────────────────────────────────────────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
MODEL  = os.environ.get(
    "BEDROCK_MODEL",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

REVIEW_PROMPTS = {
    "security": (
        "You are a senior application security engineer. "
        "Review the following git diff for security vulnerabilities such as: "
        "injection attacks (SQL, command, LDAP), broken authentication, "
        "insecure data exposure, insecure direct object references, "
        "cross-site scripting (XSS), hardcoded secrets, and path traversal.\n\n"
        "If you find a CRITICAL vulnerability that must block the merge, "
        "start your response with the exact token: CRITICAL_ISSUE_FOUND\n"
        "Otherwise start with: NO_CRITICAL_ISSUES\n\n"
        "Then provide your detailed review."
    ),
    "performance": (
        "You are a senior performance engineer specialising in Python. "
        "Review the following git diff for performance problems such as: "
        "N+1 query patterns, unbounded loops, blocking I/O in async code, "
        "unnecessary data copies, missing caching, and memory leaks.\n\n"
        "If you find a CRITICAL performance regression that must block the merge, "
        "start your response with the exact token: CRITICAL_ISSUE_FOUND\n"
        "Otherwise start with: NO_CRITICAL_ISSUES\n\n"
        "Then provide your detailed review."
    ),
    "best-practices": (
        "You are a Python code quality expert. "
        "Review the following git diff for violations of Python best practices: "
        "PEP 8 violations, missing type hints on public functions, "
        "poor error handling (bare except), mutable default arguments, "
        "functions longer than 50 lines, and missing docstrings on public APIs.\n\n"
        "If you find a CRITICAL quality issue that must block the merge, "
        "start your response with the exact token: CRITICAL_ISSUE_FOUND\n"
        "Otherwise start with: NO_CRITICAL_ISSUES\n\n"
        "Then provide your detailed review."
    ),
}


def get_diff() -> str:
    """Return the unified diff for the last commit."""
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD~1", "HEAD", "--unified=5"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return diff.strip()
    except subprocess.CalledProcessError:
        # First commit or shallow clone: diff against empty tree
        try:
            diff = subprocess.check_output(
                ["git", "show", "--unified=5", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return diff.strip()
        except subprocess.CalledProcessError:
            return ""


def review(review_type: str, diff: str) -> tuple[bool, str]:
    """Call Bedrock Claude to review the diff.
    Returns (critical_found, review_text).
    """
    if not diff:
        print(f"[AI Review:{review_type}] No diff found — skipping.")
        return False, "No changes to review."

    system_prompt = REVIEW_PROMPTS[review_type]
    user_message  = f"Git diff to review:\n\n```diff\n{diff}\n```"

    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    response = bedrock.converse(
        modelId=MODEL,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.1},
    )

    review_text: str = response["output"]["message"]["content"][0]["text"]
    critical = review_text.strip().startswith("CRITICAL_ISSUE_FOUND")
    return critical, review_text


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: ai_review.py <security|performance|best-practices>")
        sys.exit(2)

    review_type = sys.argv[1]
    if review_type not in REVIEW_PROMPTS:
        print(f"Unknown review type: {review_type}")
        sys.exit(2)

    diff = get_diff()
    print(f"\n{'=' * 60}")
    print(f"AI Code Review: {review_type.upper()}")
    print(f"Model: {MODEL}")
    print(f"Diff size: {len(diff)} chars")
    print(f"{'=' * 60}\n")

    critical, text = review(review_type, diff)

    print(text)
    print(f"\n{'=' * 60}")

    if critical:
        print(f"❌  CRITICAL issues found in [{review_type}] review — blocking merge.")
        sys.exit(1)
    else:
        print(f"✅  No critical issues in [{review_type}] review.")
        sys.exit(0)


if __name__ == "__main__":
    main()
