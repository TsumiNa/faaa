# %%
from faaa.app import FA


async def example_generate_plan():
    fa_instance = FA()
    query = "How to improve customer satisfaction?"
    plan = await fa_instance._generate_plan(query)
    print(plan)


# To run the example, you would need an event loop, for instance:
# import asyncio
# asyncio.run(example_generate_plan())
await example_generate_plan()
# %%
