group "all" {
  targets = ["qwen-0-6b", "qwen-4b", "qwen-8b", "qwen-14b"]
}

target "qwen-0-6b" {
  dockerfile = "Dockerfile"
  tags = ["justinrunpod/qwen:0.6b"]
  args = {
    MODEL_NAME = "Qwen/Qwen3-0.6B"
  }
  platforms = ["linux/amd64"]
}

target "qwen-4b" {
  dockerfile = "Dockerfile"
  tags = ["justinrunpod/qwen:4b"]
  args = {
    MODEL_NAME = "Qwen/Qwen3-4B"
  }
  platforms = ["linux/amd64"]
}

target "qwen-8b" {
  dockerfile = "Dockerfile"
  tags = ["justinrunpod/qwen:8b"]
  args = {
    MODEL_NAME = "Qwen/Qwen3-8B"
  }
  platforms = ["linux/amd64"]
}

target "qwen-14b" {
  dockerfile = "Dockerfile"
  tags = ["justinrunpod/qwen:14b"]
  args = {
    MODEL_NAME = "Qwen/Qwen3-14B"
  }
  platforms = ["linux/amd64"]
}