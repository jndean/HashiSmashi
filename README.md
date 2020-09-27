# HashiSmashi

A fun distraction.

Connect to your android device with adb and open an unsolved [hashi](https://play.google.com/store/apps/details?id=com.conceptispuzzles.hashi&hl=en) puzzle. 

The *run.py* script will:

- Grab a screenshot of the device
- Find the Hashi board, deduce its dimensions and OCR the numbers
- Solve the puzzle internally
- Send a sequence of touch inputs to the device to complete the puzzle

<p align="center">
  <img src="demo.gif">
</p>


