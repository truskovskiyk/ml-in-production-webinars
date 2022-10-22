import asyncio
from cProfile import run
import time

async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")


# async def main():
#     await count()
#     await count()
#     await count()
    

# async def main():
#     # await count()
#     # await count()
#     # await count()
#     await asyncio.gather(count(), count(), count())

async def main():
    loop = asyncio.get_event_loop()
    t1 = loop.create_task(count())
    t2 = loop.create_task(count())
    t3 = loop.create_task(count())

    await t1
    await t2
    await t3
    # await asyncio.gather(count(), count(), count())

def run_example():
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

    # s = time.perf_counter()
    # loop = asyncio.new_event_loop()
    # loop.run_until_complete(main())
    
    # elapsed = time.perf_counter() - s
    # print(f"{__file__} executed in {elapsed:0.2f} seconds.")    

if __name__ == "__main__":
    run_example()
