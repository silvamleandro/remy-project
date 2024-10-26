# 🐀 REMY Project
collaboRative intrusion dEtection system for unManned aerial vehicles swarm SecuritY

## 📖 Description

<p align="center">
  <img src="https://github.com/silvamleandro/remy-project/blob/main/imgs/remy_logo.png" width="150">
</p>

Unmanned Aerial Vehicles (UAVs), also known as drones, have become increasingly popular in various applications, including military, civil, and commercial. However, these vehicles are vulnerable to cyberattacks that can compromise security and privacy and cause physical harm. These attacks can involve signal interception, unauthorized access, data theft, and even remote control of the UAV, among others. Therefore, UAV manufacturers and users need to be aware of the threats and take appropriate measures to raise levels of security and integrity. One of the preventive measures is the use of Intrusion Detection System (IDS), which monitors system settings, data files, and network transmission to identify any abnormal behavior. Upon detection, the IDS notifies the ground control station so that a decision can be made. UAV IDS generally focus attacks on a specific data source without mentioning applying to a swarm. In this sense, this project presents the REMY.

REMY detects attacks on the UAV network and in-flight anomalies, applying machine learning techniques through supervised and unsupervised learning. The threats identified in the network are blackhole, grayhole, and flooding, and in turn, in flight are GPS spoofing and jamming. Federated Learning (FL) is also present in REMY to ensure data privacy and training collaboration between UAVs. Furthermore, geographic and physical characteristics are considered to make the IDS operation independent of the hardware.

> **PS:** The attacks on GPS were carried out using the following repository: [GPS-SDR-SIM](https://github.com/silvamleandro/gps-sdr-sim)

## 🚀 Getting Started

### Dependencies

This project requires Python version **3.10** or higher.

> **Note:** Python 3.10.3 is recommended.

### Installing

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Executing

The project is structured as follows:

```
├── libs
├── src
│   ├── flight
│   │   ├── centralized
│   │   └── distributed
│   │       ├── run.sh
│   └── network
│       ├── centralized
│       └── distributed
│           ├── run.sh
├── uav_control
│   └── pynput_flight_control.py
```

In the `libs` directory, you find the utilities needed for the system to run. The `src` directory is divided into a `centralized`, where the experiments are located in `.ipynb` files, and a `distributed` approach, which contains the FL application. he modules for _attack detection in the network_ and _anomaly identification during flight_ follow the same format.

Finally, the `uav_control` directory contains the script to control the **Parrot Bebop 2** UAV.

<br>

To execute the distributed approach, use the following command:

```bash
./run.sh
```

#### Notes:

- The results are available in the `reports` folders found throughout the project, including `.html`, `.csv`, and `.txt` files; 
- In the `v0` folder, you find the codes developed in the first part of the project. This includes the work developed during the exchange at the **MRS group**.

## 📝 Authors

<a href="https://github.com/silvamleandro/UAV_Platform/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=silvamleandro/UAV_Platform" />
</a>

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

<p align="right">
  <img src="https://github.com/silvamleandro/remy-project/blob/main/imgs/remy_mascot.png" width="300">
</p>

## ✨ Thankful
<div align="center">
    <a href="https://www.gov.br/capes/pt-br">
      <img src="https://github.com/silvamleandro/remy-project/blob/main/imgs/capes.png" width="100"/>
    </a>
    <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <a href="https://www.icmc.usp.br/">
      <img src="https://github.com/silvamleandro/remy-project/blob/main/imgs/icmc_usp.png" width="170"/>
    </a>
    <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <a href="https://www.lsec.icmc.usp.br/">
      <img src="https://github.com/silvamleandro/remy-project/blob/main/imgs/lsec_lab.png" width="100"/>
    </a>
</div>

## ⚖️ License

[MIT](https://choosealicense.com/licenses/mit/)

[⬆ Back to top](#-remy-project)<br>
