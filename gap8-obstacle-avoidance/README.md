**影像緩衝區輪替說明**<br/>
**C CODE : [nanoflownet_unquantized.c](https://github.com/NTHU-CRAZYFLIE-LAB/AI-DECK-PROJECT/blob/HSIAO/nanoflownet/gap8-obstacle-avoidance/nanoflownet_unquantized.c)**

當涉及到影像捕捉和處理時，使用兩個緩衝區來輪替影像數據是一種常見的技術。這樣做的目的是提高效率和避免數據覆蓋。下面是這個輪替過程的詳細說明：

###### **1. 緩衝區的分配**

首先，程序會分配兩個緩衝區來儲存影像數據。這兩個緩衝區分別是 `imgBuff0` 和 `imgBuff1`：

```bash
imgBuff0 = (uint8_t *)pmsis_l2_malloc((CAMERA_WIDTH * CAMERA_HEIGHT) * sizeof(uint8_t));
imgBuff1 = (uint8_t *)pmsis_l2_malloc((CAMERA_WIDTH * CAMERA_HEIGHT) * sizeof(uint8_t));
```

* `imgBuff0` 和 `imgBuff1`用來儲存從相機捕捉到的影像數據。

<br/>

###### **2. 指標的設置**

接下來，程式碼將這兩個緩衝區的地址分配給 Input_1 和 Input_2：

```bash
Input_1 = imgBuff0;
Input_2 = imgBuff1;
```

* `Input_1` 現在指向 `imgBuff0`，`Input_2` 指向 `imgBuff1`。

<br/>

###### **3. 捕捉影像**

在每次捕捉影像時，程式會將影像數據存儲到 `Input_1` 中

```bash
pi_camera_capture(&cam, Input_1, CAMERA_WIDTH * CAMERA_HEIGHT);
```

* 上述函數會將相機捕捉到的影像數據存儲到 `Input_1` 指向的緩衝區，即 `imgBuff0`

<br/>

###### **4. 處理影像**

影像數據捕捉後，程序會對其進行處理。例如，使用 CNN 模型進行計算：

```bash
nanoflownet_unquantizedCNN(Input_1, Input_2, Output_1);
```

* 在這裡，`Input_1` (即 `imgBuff0`) 中的影像數據會與 `Input_2` (即 `imgBuff1`) 中的影像數據一起被用於 CNN 模型計算。

<br/>

###### **5. 交換緩衝區**

為了準備下一次捕捉和處理，程式會交換 `Input_1` 和 `Input_2` 指向的緩衝區：

```bash
uint8_t *temp = Input_1;
Input_1 = Input_2;
Input_2 = temp;
```

* `Input_1` 現在會指向 `imgBuff1`，而 `Input_2` 則指向 `imgBuff0`。這樣在下一次捕捉影像時，新的數據會存儲在 `Input_1` 指向的緩衝區，而舊的數據則在 `Input_2` 指向的緩衝區中

<br/>

###### **6. 繼續迴圈**

上述步驟會在一個循環中重複進行：

1. 捕捉新影像並存儲在 Input_1 指向的緩衝區。
2. 使用 Input_1 和 Input_2 進行處理。
3. 交換 Input_1 和 Input_2，準備下一次操作。

<br/>

###### **總結**

這種緩衝區輪替的技術主要有以下優點：

* **避免數據覆蓋**：新捕捉的影像數據不會覆蓋尚在處理中的數據。
* **提高效率**：在影像處理的過程中，能夠同時捕捉和處理影像，避免等待時間。


