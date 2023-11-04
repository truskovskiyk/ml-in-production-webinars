import modal

stub = modal.Stub("example-get-started")

# playwright_image = modal.Image.debian_slim(python_version="3.10").run_commands(
#     "apt-get update",
#     "apt-get install -y software-properties-common",
#     "apt-add-repository non-free",
#     "apt-add-repository contrib",
#     "pip install playwright==1.30.0",
#     "playwright install-deps chromium",
#     "playwright install chromium",
# )


# @stub.function(image=playwright_image)
@stub.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@stub.local_entrypoint()
def main():
    print("the square is", square.remote(42))
