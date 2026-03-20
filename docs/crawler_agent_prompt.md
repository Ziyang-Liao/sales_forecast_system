# 爬虫 Agent 提示词：Amazon 外部数据采集

## 背景

我在做亚马逊美国站（amazon.com）**Govee 品牌**（智能家居/照明）的 SKU 级别每日销量预测。目前基于内部运营数据已达到 69.2% 的预测准确率，需要采集外部数据进一步提升。

共 59 个在售 SKU，覆盖 25 个品类。

---

## 产品列表（59个，按品类分组，仅公开信息）

### 灯带 LED Strip Lights（9个）
| 售价 | Title |
|------|-------|
| $46.8 | Govee 65.6ft RGBIC LED Strip Lights for Bedroom, Smart Strip Lights |
| $17.9 | Govee Warm White LED Strip Lights, Bright 300 LEDs, 3000K Dimmable |
| $50.0 | Govee 65.6ft RGBIC LED Strip Lights, Color Changing LED Strips |
| $40.9 | Govee RGBIC LED Strip Lights, 65.6ft Smart LED Lights for Bedroom |
| $58.0 | Govee 100ft RGBIC LED Strip Lights, Smart LED Lights for Bedroom |
| $42.3 | Govee RGBIC Alexa LED Strip Light 32.8ft, Smart WiFi LED Lights |
| $16.5 | Govee Smart RGB LED Strip Lights, 16.4ft WiFi LED Strip Lighting |
| $37.9 | Govee RGBIC Pro LED Strip Lights, 32.8ft Color Changing Smart LED |
| $32.5 | Govee 32.8ft White LED Strip Lights, 6500K Dimmable LED Light Strip |

### 温湿度计 Hygrometers & Thermometers（7个）
| 售价 | Title |
|------|-------|
| $26.5 | Govee Indoor Hygrometer Thermometer 3 Pack, Bluetooth Humidity |
| $17.2 | Govee Bluetooth Hygrometer Thermometer, Large LCD, Max/Min Records |
| $29.6 | Govee WiFi Thermometer Hygrometer H5103, Indoor Bluetooth Temperature |
| $57.8 | Govee WiFi Temperature Sensor H5179, Smart Hygrometer Thermometer |
| $43.8 | Govee WiFi Hygrometer Thermometer Sensor 3 Pack, Indoor Wireless |
| $82.0 | Govee WiFi Hygrometer Thermometer 6 Pack H5100, Indoor Wireless |
| $31.8 | Govee Hygrometer Thermometer 3Pack H5100, Mini Bluetooth Temperature |

### 屋檐灯 Permanent Outdoor Lights（7个）
| ASIN | 售价 | Title |
|------|------|-------|
| A0060 | $212.9 | Govee Permanent Outdoor Lights 2, 100ft RGBIC |
| A0108 | $53.8 | Govee 16.4ft Extension Lights for Permanent Outdoor Lights Pro |
| — | $351.9 | Govee Permanent Outdoor Lights Pro, 100ft with 60 RGBIC LED |
| — | $423.5 | Govee Permanent Outdoor Lights, Smart RGBIC Outdoor Lights |
| A0133 | $302.1 | Govee Permanent Outdoor Lights 2, 150ft RGBIC Outdoor Lights |
| A0030 | $481.7 | Govee Permanent Outdoor Lights Pro, 150ft with 90 RGBIC LED |
| A0155 | $42.3 | Govee 16.4ft Extension Lights Only for Permanent Outdoor Lights 2 |

### 智能插座 Smart Plugs（5个）
| ASIN | 售价 | Title |
|------|------|-------|
| A0052 | $22.2 | Govee Smart Plug with Energy Monitoring, WiFi Bluetooth |
| — | $21.9 | Govee Smart Plug 15A, WiFi Bluetooth Outlets 4 Pack |
| — | $11.0 | Govee Smart Plug 15A, WiFi Bluetooth Outlet 1 Pack |
| — | $18.7 | Govee Smart Plug, WiFi Plugs Work with Alexa & Google Assistant |
| — | $27.7 | Govee Dual Smart Plug 4 Pack, 15A WiFi Bluetooth Outlet |

### 球泡灯 Smart Light Bulbs（4个）
| ASIN | 售价 | Title |
|------|------|-------|
| A0005 | $27.7 | Govee BR30 Smart Light Bulbs, Works with Matter, Alexa |
| A0145 | $12.2 | Govee Smart Light Bulbs, Color Changing Light Bulb |
| — | $36.9 | Govee Smart Light Bulbs, 1200 Lumens Dimmable BR30, RGBWW |
| — | $41.1 | Govee LED Smart Light Bulbs, 1000LM Color Changing |

### TV灯带/摄像头取色 TV Backlights（5个）
| ASIN | 售价 | Title |
|------|------|-------|
| A0007 | $111.6 | Govee Envisual TV LED Backlight T2 with Dual Cameras, 11.8ft |
| A0023 | $25.9 | Govee TV LED Backlight, RGBIC Smart LED Strip for 70-80" TVs |
| — | $74.7 | Govee TV Backlight 3 Lite with Fish-Eye Correction, Sync to 75" |
| A0124 | $60.2 | Govee TV Backlight 3 Lite with Fish-Eye Correction, Sync to 40" |
| — | $13.3 | Govee TV LED Backlight Strip, RGBIC Smart LED for 40-50" TVs |

### 球泡灯串 Outdoor String Lights（3个）
| ASIN | 售价 | Title |
|------|------|-------|
| A0092 | $115.1 | Govee Smart Outdoor String Lights 2, 144ft |
| — | $37.8 | Govee Smart Outdoor String Lights H7015, 48ft RGBIC |
| — | $89.7 | Govee Smart Outdoor String Lights H7016, 96ft RGBIC |

### 其他品类（19个）
| ASIN | 品类 | 售价 | Title |
|------|------|------|-------|
| — | 筒灯 Recessed Lighting | $93.5 | Govee Smart Recessed Lighting 6 Inch |
| A0032 | 户外投影灯 Outdoor Projector | $70.3 | Govee Outdoor Projector Light, Laser and Aurora |
| — | 室内壁灯 String Downlights | $53.0 | Govee RGBIC String Downlights |
| A0055 | 壁灯 Outdoor Wall Light | $118.0 | Govee Outdoor Wall Light, 1500LM Smart RGBIC Porch |
| — | 车灯 Car LED Lights | $18.8 | Govee Car LED Strip Lights, Smart RGBIC Interior |
| A0063 | 落地灯 Floor Lamp | $109.6 | Govee Floor Lamp 2, RGBIC |
| — | 条状灯 Wall Lights | $135.6 | Govee Glide RGBIC Wall Lights, Music Wall Lights |
| — | 户外灯带 Outdoor LED Strip | $118.8 | Govee Outdoor LED Strip Lights, 98.4ft |
| A0074 | 床头灯 Table Lamp | $61.3 | Govee RGBIC Smart Table Lamp 2 |
| — | 车灯 Underglow Lights | $39.4 | Govee Underglow Car Lights, 4pcs RGBIC |
| — | 窗帘灯 Curtain Lights | $90.8 | Govee Curtain Lights, Smart LED Color Changing |
| — | 传感器 Music Sync Box | $27.6 | Govee Music Sync Box, Bluetooth Group Control |
| A0118 | 吸顶灯 Ceiling Light | $49.8 | Govee Smart Ceiling Light, RGBIC LED |
| — | 方块灯 Hexa Light Panels | $117.8 | Govee Glide Hexa Light Panels, RGBIC Hexagon |
| — | 电水壶 Electric Kettle | $49.1 | Govee Smart Electric Kettle, WiFi Variable Temperature |
| — | 新形态灯带 Gaming Lights | $66.9 | Govee RGBIC Gaming Lights, 10ft Neon Rope Lights |
| — | 烤肉温度计 Meat Thermometer | $20.8 | Govee Bluetooth Meat Thermometer |
| — | 条状灯 Glide Wall Lights | $57.9 | Govee Glide Wall Lights, RGBIC LED Light |
| — | 漏水报警器 Water Leak Detector | $40.2 | Govee Water Leak Detectors 5 Pack |

> 注：标"—"的 ASIN 未知，需通过 Title 在 Amazon 搜索匹配获取。

---

## 需要采集的数据

### 1. 品类大盘数据（最重要）

- **数据源**：Amazon Best Sellers 排行榜、Jungle Scout、Helium 10、Keepa
- **采集内容**：
  - 每个品类的 Best Sellers Rank（BSR）Top 20 产品及排名变化
  - 品类整体销量指数/热度指数
  - 品类内新品上架数量变化
- **涉及的 Amazon 小类**：
  - LED Strip Lights
  - Smart Light Bulbs
  - Outdoor String Lights
  - Permanent Outdoor Lights / Eave Lights
  - Smart Plugs & Outlets
  - Hygrometers & Thermometers
  - TV Backlights / Bias Lighting
  - Smart Ceiling Lights
  - Floor Lamps
  - Meat Thermometers
  - Water Leak Detectors
- **时间范围**：2023-01-01 ~ 2025-11-29
- **粒度**：按天，至少按周

### 2. 竞品数据

- **数据源**：Amazon 商品页面、Keepa、CamelCamelCamel
- **采集内容**：
  - 每个品类 Top 10 竞品（非 Govee）的每日售价
  - 竞品的折扣/促销活动（Coupon、Deal、Lightning Deal）
  - 竞品的评分（Rating）和评论数（Review Count）变化
  - 竞品的库存状态（是否缺货/FBA 可售）
- **重点竞品品牌**：
  - 灯带/灯具类：Philips Hue, LIFX, Wyze, Kasa (TP-Link), Sengled, Nanoleaf
  - 温湿度计类：ThermoPro, AcuRite, SensorPush
  - 智能插座类：Kasa (TP-Link), Wyze, Amazon Smart Plug, Meross
  - 烤肉温度计：ThermoPro, MEATER, Inkbird
  - 漏水报警器：Ring, YoLink, Honeywell
- **时间范围**：2023-01-01 ~ 2025-11-29
- **粒度**：按天

### 3. 搜索热度数据

- **数据源**：Google Trends（美国地区）
- **核心关键词**：
  - 灯带：`LED strip lights`, `RGBIC LED lights`, `smart LED strip`
  - 温湿度计：`hygrometer`, `indoor thermometer`, `WiFi thermometer`
  - 屋檐灯：`permanent outdoor lights`, `eave lights`, `RGBIC outdoor lights`
  - 智能插座：`smart plug`, `WiFi smart plug`, `smart outlet`
  - 球泡灯：`smart light bulb`, `color changing light bulb`
  - TV灯带：`TV backlight`, `TV LED backlight`, `bias lighting`
  - 车灯：`car LED lights`, `interior car lights`, `underglow lights`
  - 烤肉温度计：`meat thermometer`, `bluetooth meat thermometer`
  - 漏水报警器：`water leak detector`, `water alarm sensor`
  - 品牌：`Govee`, `Govee lights`
- **时间范围**：2023-01-01 ~ 2025-11-29
- **粒度**：按周

### 4. 价格历史

- **数据源**：Keepa、CamelCamelCamel
- **采集内容**：
  - 上述 59 个产品的历史售价变化（通过 ASIN 或 Title 搜索）
  - 历史 Coupon/Deal 记录
- **时间范围**：2023-01-01 ~ 2025-11-29

### 5. 宏观/季节性数据

- **数据源**：FRED（美联储经济数据）、美国商务部
- **采集内容**：
  - 美国消费者信心指数（月度）
  - 美国零售销售数据（月度）
  - 美国节假日+购物节日历：Prime Day（7月）、Prime Big Deal Days（10月）、黑五（11月第4个周五）、网一、圣诞、新年、情人节、母亲节、父亲节、返校季（8月）、Labor Day（9月）、Halloween（10月）
- **时间范围**：2023-01-01 ~ 2025-11-29

---

## 输出格式

每类数据输出为独立 CSV 文件：

**1. category_trends.csv**
```csv
date,category,bsr_top1,bsr_top10_avg,new_listings_count,category_sales_index
2025-10-01,LED Strip Lights,1523,5200,12,85.3
```

**2. competitor_data.csv**
```csv
date,category,competitor_brand,competitor_asin,price,rating,review_count,has_coupon,has_deal,in_stock
2025-10-01,LED Strip Lights,Philips Hue,B0XXXXXXXX,29.99,4.3,1256,1,0,1
```

**3. search_trends.csv**
```csv
date,keyword,google_trends_index,region
2025-10-01,LED strip lights,78,US
```

**4. price_history.csv**
```csv
date,asin,title,price,has_coupon,has_deal,deal_type
2025-10-01,A0005,Govee BR30 Smart Light Bulbs...,27.70,0,0,
```

**5. macro_data.csv**
```csv
date,consumer_confidence_index,retail_sales_yoy,is_holiday,holiday_name
2025-10-01,108.7,3.2%,0,
```

## 关键约束

1. **时间范围必须覆盖 2023-01-01 ~ 2025-11-29**
2. **地区限定美国**（amazon.com，Google Trends 选 United States）
3. **频率尽量按天**，搜索热度可按周
4. **注意反爬**：合理请求间隔、User-Agent 轮换
5. **数据要可追溯**：记录数据来源 URL 和采集时间
6. **优先级**：品类大盘 > 竞品数据 > 搜索热度 > 价格历史 > 宏观数据
7. **无 ASIN 的产品**：通过 Title 关键词在 Amazon 搜索匹配，优先匹配品牌"Govee"+产品关键词
