{
  "targets": "variance head: sigma = 0.3 m/s (quiet+GT-stopped) or 20 m/s (busy+GT-driving), target = sigma^2 per axis; ambiguous windows masked out of the noise loss. speed head: log(1 + v) with v = wheel-encoder speed at the window centre; all windows contribute",
  "lambda_speed": 1.0,
  "window_s": 1.0,
  "stride_s": 5.0,
  "sessions": 72,
  "windows": 19571,
  "deployed": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver D.pt",
  "deployed_driver": "Driver D",
  "deployed_speed_frac_all": 0.7687433209806577,
  "deployed_speed_frac_drive": 0.6840115137666957,
  "folds": [
    {
      "test_driver": "Driver A",
      "val_driver": "Driver B",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver A.pt",
      "validation_drift": 6.548356056213379,
      "test_composite": 7.967379093170166,
      "test_noise": 7.022970676422119,
      "speed_test": {
        "rmse": 7.344338370800945,
        "frac_all": 0.8575593675503399,
        "rmse_drive": 7.48115745625891,
        "frac_drive": 0.7258487133717635,
        "mean_v": 8.56423315831813
      },
      "windows": {
        "train": 13068,
        "val": 1235,
        "test": 5268
      },
      "mean": [
        1.560809229908095,
        1.5608489276419124,
        1.659842137800968,
        1.6598352399009595,
        1.6589131302246223,
        1.659028469832833
      ],
      "std": [
        3.742072872944814,
        3.7384572137737764,
        3.8240407007070654,
        3.8211627119165708,
        3.824061773202672,
        3.820746645101819
      ]
    },
    {
      "test_driver": "Driver B",
      "val_driver": "Driver D",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver B.pt",
      "validation_drift": 17.204126358032227,
      "test_composite": 10.4623441696167,
      "test_noise": 8.533985137939453,
      "speed_test": {
        "rmse": 10.395119363970224,
        "frac_all": 0.94538304292124,
        "rmse_drive": 10.73245078709973,
        "frac_drive": 0.8726792440372945,
        "mean_v": 10.995669365772878
      },
      "windows": {
        "train": 16998,
        "val": 1338,
        "test": 1235
      },
      "mean": [
        1.5638205760068786,
        1.5638544581220652,
        1.6627363884380026,
        1.6627412453220551,
        1.662057920682336,
        1.6621515545328511
      ],
      "std": [
        3.7249719714749303,
        3.722030343381065,
        3.8079039275507736,
        3.805515894831953,
        3.807890292015041,
        3.8051548786242653
      ]
    },
    {
      "test_driver": "Driver D",
      "val_driver": "Driver E",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver D.pt",
      "validation_drift": 5.587366580963135,
      "test_composite": 10.14500617980957,
      "test_noise": 9.287827491760254,
      "speed_test": {
        "rmse": 6.497995629252641,
        "frac_all": 0.7687433209806577,
        "rmse_drive": 6.733869225457954,
        "frac_drive": 0.6840115137666957,
        "mean_v": 8.452750680114379
      },
      "windows": {
        "train": 6503,
        "val": 11730,
        "test": 1338
      },
      "mean": [
        1.5709071040541098,
        1.5709526402662692,
        1.6695013619220844,
        1.6694716636047726,
        1.669959316859542,
        1.6699415461836553
      ],
      "std": [
        3.671090613444354,
        3.6706131736138117,
        3.757801961682402,
        3.7573475608629496,
        3.7575012823966696,
        3.7570057454233394
      ]
    },
    {
      "test_driver": "Driver E",
      "val_driver": "Driver A",
      "checkpoint": "/home/zenodefault/code/gati-nav/python/ml/weights/noisenet_fold_Driver E.pt",
      "validation_drift": 12.468857765197754,
      "test_composite": 8.614649772644043,
      "test_noise": 2.537672996520996,
      "speed_test": {
        "rmse": 15.779441001490556,
        "frac_all": 1.276933910108356,
        "rmse_drive": 18.243428832824456,
        "frac_drive": 1.0925334781012894,
        "mean_v": 12.357288718373505
      },
      "windows": {
        "train": 2573,
        "val": 5268,
        "test": 11730
      },
      "mean": [
        1.5664367520922875,
        1.5665296256962382,
        1.6651345832457585,
        1.6649924039879596,
        1.666055844365041,
        1.665978155715523
      ],
      "std": [
        3.676537630125035,
        3.6763324400475703,
        3.7640069165523777,
        3.763970887367709,
        3.7634533425310073,
        3.7633793105208193
      ]
    }
  ],
  "elapsed_s": 295.4
}
