## How to dump Aub File for cross_runner.exe
1. donwload rasty in [this link](https://fictional-dollop-3y8ye44.pages.github.io/CHANGELOG.html), you should always match the version with the driver you are using.
1. launch Rasty on any  device you have
2. load .rlsetting file, here is one [rasty](./rasty_lnl_crossrunner.rlsettings) setting file you can refer
3. check the settings you are interested is correct:  test case folder and command line; aub dump folder; user_defined_driver
4. launch, it will run the app
## how to dump OCL aub file?

IR driver, set regKey below, then run the command line, it will dump the aubfile . OCL interpreter cannot be used in this case.
```
set SetCommandStreamReceiver=3
set PrintDebugSettings=1
```

# Getting ORT Core API Verbose logs using ORT GenAI
```
import os
os.environ["ORTGENAI_ORT_VERBOSE_LOGGING"] = "1" 
```