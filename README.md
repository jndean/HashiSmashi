# HashiSmashi

A fun distraction from writing [AlphaLogger](https://github.com/jndean/AlphaLogger).

Connect to your android device with adb and open an unsolved [hashi](https://play.google.com/store/apps/details?id=com.conceptispuzzles.hashi&hl=en) puzzle. 

The *run.py* script will:

- Grab a screen cap of the device
- Find the hashi board and perform OCR
- Solve the puzzle internally
- Send a sequence of touch inputs to the device to complete the puzzle