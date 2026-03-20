"""Endpoint and credential configuration for the Medical AI POC."""

# Hardcoded for this POC. A configuration UI is coming soon.
ENDPOINT_NAME = "ai-poc-endpoint"
REGION = "us-east-1"
CONTENT_TYPE = "application/json"

# S3 bucket for volume uploads
VOLUME_S3_BUCKET = "ai-poc-sagemaker-endpoint-706262411476"
VOLUME_S3_PREFIX = "volumes/"

# Background job polling (Merlin submit+poll pattern)
BACKGROUND_POLL_INTERVAL = 5    # seconds between poll requests
BACKGROUND_POLL_TIMEOUT = 300   # max seconds to wait for result

# AWS credentials are read from environment variables.
# Set these before launching Voilà:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-1
