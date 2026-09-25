PixelPaws example project
=========================

Six 60-second clips from one formalin experiment and a key file. Nothing has
been computed on them: run the pipeline yourself and watch each step.

Three mice got vehicle and three got a drug before 2% formalin in the left
hind paw. The clips start 1 min 40 s into the recording (the early phase,
when licking is heaviest), filmed from below at 60 fps with PawCapture, so
every clip carries its mm/pixel calibration and the Locomotion tab reports
centimeters.

  videos/            mouse1_veh, mouse2_veh, mouse3_veh,
                     mouse4_drug, mouse5_drug, mouse6_drug (.mp4)
  key.csv            Subject -> Treatment (Vehicle / Drug)
  PixelPaws_project.json

1. Open it: click "Open the example project" in the setup window that
   opens when PixelPaws starts, or File > Load Project and pick this folder.
2. On the Quick Start tab the six clips show "needs analysis". Select them,
   leave the steps ticked, press Run pipeline. Pose tracking takes about a
   minute per clip on a GPU (longer on CPU), then features, the Core 8
   classifiers and paw contour analysis run on their own.
3. When it finishes, the Analyze tabs are filled in from results/ and
   key.csv. Review scoring plays a clip with the classifier calls on it: the
   vehicle clips have a lot of left-paw licking, the drug clips almost none.

Three animals per group is enough to show the graphs, not to test an effect.
