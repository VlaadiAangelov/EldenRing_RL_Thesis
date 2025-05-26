import cv2
import numpy as np
import time
import pytesseract 
import os
import csv
import datetime
import re
import easyocr



'''Detecting if the enemy is damaged in PvP'''
def extract_hp_ratios_from_frame(frame):
    """Extracts player and boss HP ratios from the frame using HSV masking and returns (player_hp_ratio, boss_hp_ratio)."""

    # --- Player HP ---
    hp_image = frame[90:107, 200:370]
    hsv_hp = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)
    lower_hp1 = np.array([0, 60, 60])
    upper_hp1 = np.array([15, 255, 255])
    mask_hp1 = cv2.inRange(hsv_hp, lower_hp1, upper_hp1)
    lower_hp2 = np.array([160, 60, 60])
    upper_hp2 = np.array([180, 255, 255])
    mask_hp2 = cv2.inRange(hsv_hp, lower_hp2, upper_hp2)
    mask_hp = cv2.bitwise_or(mask_hp1, mask_hp2)
    player_hp_ratio = np.count_nonzero(mask_hp) / (mask_hp.shape[0] * mask_hp.shape[1])
    player_hp_ratio = min(player_hp_ratio + 0.02, 1.0)  # add tolerance

    # --- Boss HP ---
    boss_hp_image = frame[800:830, 430:1370]
    hsv_boss = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)
    lower_boss = np.array([0, 130, 0])
    upper_boss = np.array([255, 255, 255])
    mask_boss = cv2.inRange(hsv_boss, lower_boss, upper_boss)
    projection = np.sum(mask_boss, axis=0)
    filled_columns = np.count_nonzero(projection > 0)
    boss_hp_ratio = filled_columns / mask_boss.shape[1]
    boss_hp_ratio = min(boss_hp_ratio + 0.01, 1.0)  # add tolerance

    return player_hp_ratio, boss_hp_ratio


class EldenReward:
    '''Reward Class'''


    '''Constructor'''
    def __init__(self, config):
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]        #Setting the path to pytesseract.exe
        self.GAME_MODE = config["GAME_MODE"]
        self.DEBUG_MODE = config["DEBUG_MODE"]
        self.DEBUG_MODE2 = config["DEBUG_MODE2"]
        self.max_hp = config["PLAYER_HP"]                             #This is the hp value of your character. We need this to capture the right length of the hp bar.
        self.prev_hp = 1.0     
        self.curr_hp = 1.0
        self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
        self.numeric_hits_this_episode = 0
        self.skip_ocr_frame = True  # toggle flag to skip every other frame
        self.time_since_dmg_taken = time.time()
        self.death = False
        self.max_stam = config["PLAYER_STAMINA"]                     
        self.curr_stam = 1.0
        self.curr_boss_hp = 1.0
        self.total_numeric_damage_reward = 0
        self.time_since_boss_dmg = time.time() 
        self.time_since_pvp_damaged = time.time()
        self.time_alive = time.time()
        self.boss_death = False    
        self.game_won = False
        self.consecutive_hits = 0
        self.last_boss_hit_time = time.time()    
        self.death_counter = 200
        self.LOG_FILE_PATH = "death_logs.csv"
        self.last_dodge_forward = time.time()
        self.prev_action = None
        self.last_action = None
        self.prev_boss_hp_image = None
        self.prev_player_hp = 1.0
        self.prev_boss_hp_ratio = 1.0
        self.last_visible_damage = None  # value last seen clearly
        self.last_applied_damage = None  # value last used for reward
        self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
        self.numeric_hits_this_episode = 0
        self.skip_ocr_frame = True  # toggle flag to skip every other frame
        self.boss_hp_history = []
        self.boss_damage_window = 5
        self.image_detection_tolerance_hp = 0.07
        self.image_detection_tolerance = 0.04           #The image detection of the hp bar is not perfect. So we ignore changes smaller than this value. (0.02 = 2%)

    '''Detecting the current player hp'''
    def get_current_hp_old(self, frame):
        HP_RATIO = 0.403                                                        #Constant to calculate the length of the hp bar
        hp_image = frame[42:51, 74:221]                                        #Cut out the hp bar from the frame
        if self.DEBUG_MODE: self.render_frame(hp_image)
        
        lower = np.array([0,90,75])                                             #Filter the image for the correct shade of red
        upper = np.array([150,255,125])                                         #Also Filter
        hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)                         #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
        if self.DEBUG_MODE: self.render_frame(mask)

        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
        curr_hp = len(matches) / (hp_image.shape[1] * hp_image.shape[0])        #Calculating percent of white pixels in the mask (current hp in percent)

        curr_hp += 0.02         #Adding +2% of hp for color noise

        if curr_hp >= 0.96:     #If the hp is above 96% we set it to 100% (also color noise fix)
            curr_hp = 1.0

        cv2.imwrite("debug_hp.png", hp_image)
        cv2.imwrite("debug_mask_hp.png", mask)

        if self.DEBUG_MODE: print('💊 Health: ', curr_hp)
        return curr_hp
    
    def get_current_hp(self, frame):
        hp_image = frame[90:107, 200:370]  # Crop HP bar
        if self.DEBUG_MODE:
            cv2.imwrite("debug_hp_bar.png", hp_image)

        hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)

        if self.DEBUG_MODE:
            mean_hsv = cv2.mean(hsv)[:3]
            print(f"🔬 Mean HSV of HP bar: {mean_hsv}")

        # Filter tuned to red (works best for Elden Ring HP bars)
        lower = np.array([0, 60, 60])     # narrow lower bound for red
        upper = np.array([15, 255, 255])  # allowing light-to-deep reds

        mask1 = cv2.inRange(hsv, lower, upper)

        # Red wraps around in HSV — handle 160-180 too
        lower2 = np.array([160, 60, 60])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

        filled_pixels = np.count_nonzero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        curr_hp = filled_pixels / total_pixels
        curr_hp = min(curr_hp + 0.02, 1.0)

        if self.DEBUG_MODE:
            print(f"💊 HP %: {curr_hp:.2f}")
            cv2.imwrite("debug_mask_hp.png", mask)

        return curr_hp
    
    def extract_boss_damage_number(self, frame):
        number_crop = frame[770:810, 1300:1370]

        # 🧠 Your working preprocessing pipeline:
        gray = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        upscaled = cv2.resize(blurred, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        if self.DEBUG_MODE2:
            timestamp = str(time.time()).replace('.', '_')
            cv2.imwrite(f"debug_live_crop_{timestamp}.png", number_crop)


        # 🧠 Use cached EasyOCR reader
        results = self.easyocr_reader.readtext(upscaled, detail=0)

        if self.DEBUG_MODE:
            timestamp = str(time.time()).replace('.', '_')
            cv2.imwrite(f"debug_easyocr_crop_{timestamp}.png", upscaled)
            print("🧠 EasyOCR result raw:", results)

        if not results:
            return None

        text = results[0]
        digits_only = re.sub(r'\D', '', text)

        if digits_only == "":
            return None

        return int(digits_only)


    def log_death(self, total_reward, cause, total_damage, steps):
        self.death_counter += 1
        file_exists = os.path.exists(self.LOG_FILE_PATH)

        with open(self.LOG_FILE_PATH, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["death_number", "total_reward","total_damage", "death_cause", "steps"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "death_number": self.death_counter,
                "total_reward": round(total_reward, 3),
                "total_damage": total_damage,
                "death_cause": cause,
                "steps": steps
            })

    '''Detecting the current player stamina'''
    def get_current_stamina_old(self, frame):
        STAM_RATIO = 3.0                                                        #Constant to calculate the length of the stamina bar
        stam_image = frame[63:72, 74:280]                                       #Cut out the stamina bar from the frame
        if self.DEBUG_MODE: self.render_frame(stam_image)

        lower = np.array([6,52,24])                                             #This filter really inst perfect but its good enough bcause stamina is not that important
        upper = np.array([74,255,77])                                           #Also Filter
        hsv = cv2.cvtColor(stam_image, cv2.COLOR_RGB2HSV)                       #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
        if self.DEBUG_MODE: self.render_frame(mask)

        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
        self.curr_stam = len(matches) / (stam_image.shape[1] * stam_image.shape[0]) #Calculating percent of white pixels in the mask (current stamina in percent)

        self.curr_stam += 0.02                                                  #Adding +2% of stamina for color noise
        if self.curr_stam >= 0.96:                                              #If the stamina is above 96% we set it to 100% (also color noise fix)
            self.curr_stam = 1.0 

        if self.DEBUG_MODE: print('🏃 Stamina: ', self.curr_stam)
        return self.curr_stam
    
    def get_current_stamina(self, frame):
        stam_image = frame[120:137, 200:463]  # Crop stamina bar
        if self.DEBUG_MODE:
            cv2.imwrite("debug_stamina_bar.png", stam_image)

        hsv = cv2.cvtColor(stam_image, cv2.COLOR_RGB2HSV)

        if self.DEBUG_MODE:
            mean_hsv = cv2.mean(hsv)[:3]
            print(f"🔬 Mean HSV of stamina bar: {mean_hsv}")

        # Tighter bounds for stamina green
        lower = np.array([35, 60, 60])   # hue-sat-val tighter around green
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Count white pixels
        filled_pixels = np.count_nonzero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        curr_stam = filled_pixels / total_pixels

        # Add noise compensation
        curr_stam = min(curr_stam + 0.02, 1.0)

        if self.DEBUG_MODE:
            print(f"🏃 Stamina %: {curr_stam:.2f}")
            cv2.imwrite("debug_stamina_mask.png", mask)

        return curr_stam
    

    '''Detecting the current boss hp'''
    def get_boss_hp_old(self, frame):
        boss_hp_image = frame[515:530, 240:860]                                #cutting frame for boss hp bar (always same size)
        if self.DEBUG_MODE: self.render_frame(boss_hp_image)

        lower = np.array([0,130,0])                                             #Filter the image for the correct shade of green
        upper = np.array([255,255,255])
        hsv = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)                    #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)
        if self.DEBUG_MODE: self.render_frame(mask)


        #self.render_frame(boss_hp_image)
        #self.render_frame(mask)


        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
        boss_hp = len(matches) / (boss_hp_image.shape[1] * boss_hp_image.shape[0])  #Calculating percent of white pixels in the mask (current boss hp in percent)
        
        #same noise problem but the boss hp bar is larger so noise is less of a problem

        if self.DEBUG_MODE: print('👹 Boss HP: ', boss_hp)

        return boss_hp
    
    def get_boss_hp(self, frame):
        boss_hp_image = frame[800:830, 430:1370]  # Crop boss HP bar
        if self.DEBUG_MODE:
            self.render_frame(boss_hp_image)
            cv2.imwrite("debug_boss_bar.png", boss_hp_image)

        # Convert to HSV and mask green
        hsv = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 130, 0])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Horizontal projection
        projection = np.sum(mask, axis=0)
        filled_pixels = np.count_nonzero(projection > 0)
        max_pixels = mask.shape[1]

        boss_hp = filled_pixels / max_pixels
        boss_hp = min(boss_hp + 0.01, 1.0)  # add noise tolerance

        if self.DEBUG_MODE:
            print(f"👹 Boss HP %: {boss_hp:.2f}")
            self.render_frame(mask)
            cv2.imwrite("debug_boss_mask.png", mask)

        return boss_hp
        

    def detect_boss_damaged(self, frame):
        # 🟥 Crop the boss HP bar area (confirmed region)
        cut_frame = frame[585:600, 285:1050]

        # ✅ Safety check to prevent OpenCV crash
        if cut_frame is None or cut_frame.size == 0:
            print("❌ Empty boss bar crop — skipping boss damage check")
            return False

        # 🎨 HSV filter (adjustable if detection is off)
        lower = np.array([20, 150, 100])
        upper = np.array([40, 255, 255])
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        matches = np.argwhere(mask == 255)

        # 🧪 Optional debug
        if self.DEBUG_MODE:
            print("🎯 Boss damage pixel matches:", len(matches))
            # Uncomment to see the mask and cropped bar
            # cv2.imshow("Boss HP Region", cut_frame)
            # cv2.imshow("Boss HP Mask", mask)
            # cv2.waitKey(1)

        # ✅ Detection threshold (30 = minimal pixel change)
        #print("🎯 Boss damage pixel matches:", len(matches))
        #return len(matches) > 30
    
    def detect_boss_kill_quote(self, frame):
        # Crop the region where the subtitle appears
        subtitle_area = frame[875:910, 800:1050]  # Adjust as needed
        subtitle_area = cv2.resize(subtitle_area, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        hsv = cv2.cvtColor(subtitle_area, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 0, 75])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        text = pytesseract.image_to_string(mask, lang='eng', config='--psm 6 --oem 3').strip()

        # if self.DEBUG_MODE:
        #     print("🧠 Subtitle OCR:", repr(text))

        # Save the cropped subtitle area for every scan
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #cv2.imwrite(f"quote_scan_{timestamp}.png", subtitle_area)

        text = text.lower()
        return text.strip() != ""

    '''Detecting if the duel is won in PvP'''
    def detect_win(self, frame):
        cut_frame = frame[730:800, 550:1350]
        lower = np.array([0,0,75])                  #Removing color from the image
        upper = np.array([255,255,255])
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        pytesseract_output = pytesseract.image_to_string(mask,  lang='eng',config='--psm 6 --oem 3') #reading text from the image cutout
        game_won = "Combat ends in your victory!" in pytesseract_output or "combat ends in your victory!" in pytesseract_output             #Boolean if we see "combat ends in your victory!" on the screen
        return game_won
    

    '''Debug function to render the frame'''
    def render_frame(self, frame):
        cv2.imshow('debug-render', frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

 
    '''Update function that gets called every step and returns the total reward and if the agent died or the boss died'''
    def update(self, frame, first_step, current_action, step_count, total_timesteps, boss_hp_crop, player_state):
        #📍 1 Getting current values
        #📍 2 Hp Rewards
        #📍 3 Boss Rewards
        #📍 4 PvP Rewards
        #📍 5 Total Reward / Return
        survival_reward = 0
        # damage_reward = 0
        # death_penalty = 0

        '''📍1 Getting/Setting current values'''
        curr_player_hp, curr_boss_hp = extract_hp_ratios_from_frame(frame)
        self.curr_stam = self.get_current_stamina(frame)            
        if first_step:
            self.time_since_dmg_taken = time.time() - 10

        # --- Boss HP Damage Detection (Improved) ---
        self.boss_hp_history.append(curr_boss_hp)
        if len(self.boss_hp_history) > self.boss_damage_window:
            self.boss_hp_history.pop(0)

        boss_damaged = False
        if len(self.boss_hp_history) >= self.boss_damage_window:
            oldest = self.boss_hp_history[0]
            newest = self.boss_hp_history[-1]
            if oldest - newest > self.image_detection_tolerance:
                boss_damaged = True
                print("🎯 Smoothed boss HP drop detected over history window")


        # --- Player damage detection ---
        player_damaged = False
        if self.prev_player_hp - curr_player_hp > self.image_detection_tolerance_hp and step_count >= 3:
            player_damaged = True
            print("💔 Detected player HP drop")

        self.prev_player_hp = curr_player_hp

        quote_detected = self.detect_boss_kill_quote(frame)
        self.death = False
        if curr_player_hp <= 0.02:
            self.death = True
            curr_player_hp = 0.0
            if quote_detected:
                print("💀 Death by boss — quote detected.")
                hp_reward = -150
            else:
                print("🕳️ Death by fall — no quote found.")
                hp_reward = -200
        else:
            if player_damaged and step_count >= 3:
                hp_reward = -40
            else:
                hp_reward = 0

        self.boss_death = False
        if self.GAME_MODE == "PVE":
            if curr_boss_hp <= 0.01:
                self.boss_death = True

        '''📍2 Hp Rewards'''
        # time_since_taken_dmg_reward = 0
        # if time.time() - self.time_since_dmg_taken > 5:
        #     time_since_taken_dmg_reward = 10

        self.prev_hp = curr_player_hp

        '''📍3 Boss Rewards'''
        boss_dmg_reward = 0
        if self.GAME_MODE == "PVE":
            if self.boss_death:
                hp_reward = -200
            elif boss_damaged:
                current_time = time.time()
                if current_time - self.last_boss_hit_time > 5:
                    self.consecutive_hits = 0
                self.consecutive_hits += 1
                self.last_boss_hit_time = current_time
                boss_dmg_reward = 50 + (self.consecutive_hits * 10)
                self.time_since_boss_dmg = current_time
                if self.DEBUG_MODE:
                    print(f"⚔️ Combo hit! Count: {self.consecutive_hits}, Reward: {boss_dmg_reward}")

        numeric_damage_reward = 0
        numeric_damage = None
        self.skip_ocr_frame = not self.skip_ocr_frame
        if not self.skip_ocr_frame:
            numeric_damage = self.extract_boss_damage_number(frame)
        if numeric_damage is not None and numeric_damage != self.last_applied_damage:
            self.numeric_hits_this_episode += 1
            numeric_damage_reward = self.numeric_hits_this_episode * 50
            self.total_numeric_damage_reward += numeric_damage_reward
            print(f"💥 Hit #{self.numeric_hits_this_episode}: +{numeric_damage_reward} (OCR: {numeric_damage})")
            self.last_applied_damage = numeric_damage

        survival_bonus = max(0.1, 0.948 - (total_timesteps / 100000))
        survival_reward += survival_bonus

        if boss_damaged and step_count >= 3:
            boss_dmg_reward = 50
        else:
            boss_dmg_reward = 0

        '''📍5 Total Reward / Return'''
        total_reward = hp_reward + survival_reward + numeric_damage_reward
        total_reward = round(total_reward, 3)
        if self.boss_death:
            self.log_death(total_reward, "WIN", self.total_numeric_damage_reward, step_count)
        elif self.death:
            cause = "Boss" if quote_detected else "Fall"
            self.log_death(total_reward, cause, self.total_numeric_damage_reward, step_count)

        self.prev_action = self.last_action
        self.last_action = current_action

        return total_reward, self.death, self.boss_death, self.game_won


'''Testing code'''
if __name__ == "__main__":
    env_config = {
        "PYTESSERACT_PATH": r"C:\Users\Legion\AppData\Local\Programs\Tesseract-OCR",    # Set the path to PyTesseract
        "MONITOR": 1,           #Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,    #Renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",     #PVP or PVE
        "BOSS": 1,              #1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "BOSS_HAS_SECOND_PHASE": True,  #Set to True if the boss has a second phase (only for PVE)
        "PLAYER_HP": 460,      #Set the player hp (used for hp bar detection)
        "PLAYER_STAMINA": 96,  #Set the player stamina (used for stamina bar detection)
        "DESIRED_FPS": 24       #Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
    }
    reward = EldenReward(env_config)

    IMG_WIDTH = 1280                                #Game capture resolution
    IMG_HEIGHT = 720 

    import mss
    sct = mss.mss()
    monitor = sct.monitors[1]
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
    frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]    #cut the frame to the size of the game

    reward.update(frame, True)
    time.sleep(1)
    reward.update(frame, False)

        