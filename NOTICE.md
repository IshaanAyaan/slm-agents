# Third-party code

The `openharness/` directory is a small vendored subset of the OpenHarness
project (MIT license), retained so the harnesses in this repo run against the
exact same file tools that the published experiments used. It contains only
the tool base classes, the grep, glob, and read file tools, the conversation
message types, and the token usage dataclass. Everything else in this
repository is original work for the SLM plus co-designed harness study.

The benchmark task generator additionally reads the source trees of installed
pip packages (requests, urllib3, flask, click, rich, httpx) at run time to
build navigation tasks. Their code is never redistributed here.
