{
  "target": "sigma = 0.3 m/s (quiet+GT-stopped) or 20 m/s (busy+GT-driving), target = sigma^2 per axis; ambiguous windows dropped; GT speed is a training label only",
  "drop": "windows with GT speed in (V_STOP_MAX, V_DRIVE_MIN) or vibration inconsistent with the motion label",
  "window_s": 1.0,
  "stride_s": 5.0,
  "sessions": 32,
  "windows": 9060,
  "deployed": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver D.pt",
  "deployed_driver": "Driver D",
  "folds": [
    {
      "test_driver": "Driver A",
      "val_driver": "Driver D",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver A.pt",
      "validation_drift": 9.105335235595703,
      "test_drift": 8.009560585021973,
      "windows": {
        "train": 5794,
        "val": 488,
        "test": 2778
      },
      "mean": [
        1.5671032205980862,
        1.5670776493882719,
        1.666044325003348,
        1.666014712005058,
        1.6644404053740682,
        1.6646595050749342
      ],
      "std": [
        3.7555995480821918,
        3.7513178505238236,
        3.836282649929733,
        3.8328336052044696,
        3.837093589903385,
        3.8330913095469428
      ]
    },
    {
      "test_driver": "Driver D",
      "val_driver": "Driver E",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver D.pt",
      "validation_drift": 2.1234395503997803,
      "test_drift": 12.33761978149414,
      "windows": {
        "train": 3340,
        "val": 5232,
        "test": 488
      },
      "mean": [
        1.5709912210613912,
        1.5711212424375696,
        1.669570781672869,
        1.6695566263708772,
        1.6700570577836213,
        1.6700485064949298
      ],
      "std": [
        3.681377616930285,
        3.6807566052210747,
        3.767456857749173,
        3.7669361412806683,
        3.7674972828285154,
        3.7667937944440726
      ]
    },
    {
      "test_driver": "Driver E",
      "val_driver": "unknown",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver E.pt",
      "validation_drift": 6.000876426696777,
      "test_drift": 2.8026580810546875,
      "windows": {
        "train": 3266,
        "val": 562,
        "test": 5232
      },
      "mean": [
        1.5786589718230044,
        1.5788278411854761,
        1.6771743377583823,
        1.6771447290100283,
        1.6772867528133464,
        1.677289257432109
      ],
      "std": [
        3.6716189729495476,
        3.6710040811907825,
        3.7581869470892038,
        3.7578182746051745,
        3.7585365715378907,
        3.7579775880672397
      ]
    },
    {
      "test_driver": "unknown",
      "val_driver": "Driver A",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_unknown.pt",
      "validation_drift": 9.04328441619873,
      "test_drift": 6.3744730949401855,
      "windows": {
        "train": 5720,
        "val": 2778,
        "test": 562
      },
      "mean": [
        1.5714310461143641,
        1.5714256433786746,
        1.6703401740210915,
        1.6703015374519619,
        1.6684957465783485,
        1.66872410410563
      ],
      "std": [
        3.751099436893703,
        3.7467695717611442,
        3.831976946444878,
        3.8285718908803887,
        3.8329737154158825,
        3.8290059098621168
      ]
    }
  ],
  "elapsed_s": 131.4
}
