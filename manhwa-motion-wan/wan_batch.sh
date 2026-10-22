#!/bin/bash
# All 38 Wan 2.2 shots of the chapter, with the exact prompts (batch 1 read back from each
# video's embedded ComfyUI graph). Each: 81 frames at 16 fps (5 s),
# seed 1, one take. Needs ComfyUI running (python main.py --lowvram) and the panels copied into
# ComfyUI/input/. Set COMFY below. Sizes are multiples of 16 near the panel's shape:
#   tall 1024x1344 -> 560x736,  square-ish 1024x896 -> 640x560,  wide 1024x448 -> 976x432.
# Resumable: a shot with a finished mp4 in ./wan is skipped.
set -u
COMFY=../ComfyUI
STYLE="Hand-drawn manhwa illustration, clean black ink lines, flat cel colour."
mkdir -p wan
wan() { # name panel.png W H prompt
  [ -f "wan/$1.mp4" ] && return
  echo "=== $1"
  python wan_i2v.py "$2" "$5 $STYLE" -W $3 -H $4 --frames 81 --seed 1 --prefix "video/wan_$1" --comfy "$COMFY" | tail -2
  cp "$(ls -t $COMFY/output/video/wan_$1_*.mp4 | head -1)" "wan/$1.mp4"
}
# batch 1 (2026-09-28)
wan lirazel 34_c19_lirazel.png 560 736 "An elf woman with long silver hair stands among white lilies beside a lake. Her hair and the long lilac sleeves of her dress drift sideways in a light breeze, the lily petals sway, she blinks once and her lips part slightly. The camera pushes in slowly."
wan forge   06_c06_forge.png   560 736 "An old witch in a ragged green cloak raises both arms higher, fingers spread; the green flames leap and flicker; orange sparks drift upward. A young man in dark robes stands still at the left, holding a black stone. The camera is still."
wan roar    25_f10_roar.png    560 736 "A forest dragon made of bark and moss throws its head back and roars, its jaw opening wider; birds burst from the branches and fly off; pine needles shake. The camera shakes slightly."
wan fields  10_c10_fields.png  976 432 "A rider on a dark horse gallops through a golden wheat field toward a tall curtain of pale blue-green light; the wheat bends in the wind; the camera tracks alongside the rider."
# batch 2 (2026-09-29): motion a camera move cannot fake, mostly without a peak expression
wan dawn     09_c09_dawn.png     640 560 "A young man in dark robes rides a grey horse at a slow walk down a dirt road toward the rising sun; the horse's head nods with each step; his long black hair and robe stir in the wind; the sky glows. The camera follows slowly from behind."
wan twilight 11_c11_twilight.png 560 736 "A young man in dark robes reaches out and touches a tall wall of pale blue-green light; ripples spread out from his fingertips across the light; the grass sways at his feet. The camera is still."
wan claws    22_f07_claws.png    640 560 "A forest dragon made of bark and moss crouches low and drags its claws through the earth; clods of dirt and grass fly up; moss on its back shivers. The camera is still."
wan draw     28_c13_draw.png     560 736 "A young man kneels in the grass holding a sword upright before his face; green runes along the blade pulse brighter and dimmer; his long black hair lifts in the wind; he stays calm and focused. The camera slowly pushes in."
wan meet     35_c20_meet.png     640 560 "A young man in dark robes slowly reaches out his open hand toward an elf woman in a lilac dress; she turns her head toward him; her long silver hair moves in a light breeze; white lilies sway. The camera is still."
wan flee     37_c22_flee.png     560 736 "A young man in dark robes and an elf woman in a lilac dress run together across a meadow, hand in hand; her silver hair and dress stream behind her; flower petals scatter. The camera tracks with them."
wan mist     13_s1.png           976 432 "A forest of tall ivy-covered trees; pale mist drifts slowly between the trunks; light shafts shimmer; leaves tremble. The camera slowly pushes forward along the path."
# the rest of the chapter (2026-09-30): the other 27 panels, so every panel has a shot
#   more sizes: 16:9 848x480, strips 1120x352 / 1184x320 / 1280x288, tall 480x816
wan erl       01_c01_erl.png          976 432 "A young man in dark robes walks down a dirt path between green vineyards toward the viewer; a stone castle on the hill behind him; his robe and long black hair stir in the breeze; the river glints; leaves sway. The camera slowly pulls back."
wan lord      02_c02_lord.png         640 560 "An old lord with a long white beard sits in a carved chair and speaks, gesturing with one open hand; a young man in dark robes kneels before him with his head bowed, listening; dust drifts in the light from the doorway. The camera is still."
wan sword     03_c03_sword.png        640 560 "A young man in dark robes lifts an old sword in a black scabbard down from iron hooks on a white wall, both hands on it, looking at it; his long black hair moves slightly. The camera slowly pushes in."
wan heath     04_c04_heath.png        976 432 "A young man in dark robes carrying a sword climbs a windswept grassy hill toward a small cottage on the top; the long grass ripples in the wind; his robe and hair blow; clouds drift. The camera tracks slowly alongside."
wan door      05_c05_door.png         640 560 "An old witch with glowing green eyes and a ragged cloak stands in a cottage doorway holding a lantern; the lantern flame flickers; she looks up at a young man in dark robes seen from behind in the foreground; her grey hair stirs. The camera is still."
wan runes     07_c07_runes.png       1120 352 "An old finger presses along a steel sword blade; green runes light up one after another along the blade; green sparks and smoke drift upward. The camera slowly pans along the blade."
wan offer     08_c08_offer.png        480 816 "An old witch with glowing green eyes holds out a sword across both palms; the green runes along the blade glow brighter; green smoke curls upward; a small green fire burns at her feet; the grass sways. The camera slowly pushes in."
wan elfland   12_c12_elfland.png      480 816 "A young man in dark robes, seen from behind, stands before a glowing valley of pale blue mountains and a crystal palace; glowing purple and blue flowers sway around him; light shimmers over the valley. The camera slowly moves forward past him."
wan vines     14_s2.png               480 816 "A huge ivy-covered tree trunk looms overhead; twisting vines creep and tighten around it; a young man in dark robes at the bottom raises an arm against them; leaves fall; light flickers through the branches. The camera slowly tilts down."
wan slice     15_s3.png               640 560 "A young man in dark robes swings a sword and slices through a thick ivy vine; the cut vine and leaves fly apart; his hair whips; speed lines. The camera is still."
wan arch      16_s4.png               560 736 "A young man in dark robes stands in an arch of ivy, holding a sword at his side, looking up; the ivy leaves tremble; his robe and hair move in the breeze; pale light drifts through the mist. The camera slowly pulls back."
wan clearing  17_f01_clearing.png     976 432 "A sunlit forest clearing ringed by tall pines; low mist drifts across the grass; light shafts shimmer; the grass sways; branches move gently. The camera slowly pushes forward."
wan turn      18_f03_turn.png         640 560 "A young man in dark robes turns sharply, gripping the sword at his hip, eyes alert; his long black hair swings; mist drifts behind him. The camera is still."
wan reveal    19_f04_reveal.png       480 816 "A huge forest dragon made of bark and moss, with antlers, slowly raises and turns its head; its amber eye glows; ferns sway; sunlight flickers through the leaves. The camera slowly tilts up."
wan eyes      20_f05c_eyes_sliver.png 1280 288 "Close-up of a young man's eyes; a bead of sweat runs down his brow; his eyes narrow and shift to the side; strands of black hair move. The camera is still."
wan ots       21_f06a_ots.png         560 736 "Over the shoulder of a young man in dark robes, a forest dragon made of bark and moss leans toward him, jaws open, amber eyes glaring; its mossy mane bristles; mist drifts. The camera is still."
wan cut       23_f08_cut.png          640 560 "A young man in dark robes slashes a sword across the frame; the blade flashes with light; his robe and hair whip; speed lines burst outward. The camera shakes slightly."
wan strike    24_f06c_s812.png        848 480 "A young man in dark robes strikes a forest dragon's snout with his sword; the blade glances off; the dragon recoils with a spray of dark red blood; moss flies. The camera shakes slightly."
wan land      26_f11_land.png         640 560 "A young man in dark robes lands in a low crouch on the grass, one hand on the ground, sword in the other hand; grass and dust scatter; his hair settles. The camera is still."
wan standoff  27_f12b_standoff.png    848 480 "A forest dragon made of bark and moss and a young man in dark robes face each other across a sunlit clearing; the dragon lowers its head and its tail sways; the young man shifts his stance, sword ready; the grass sways. The camera slowly pans."
wan finale    29_c14_finale.png       848 480 "A young man in dark robes swings a sword with a glowing green blade in a wide arc at a forest dragon; a bright green trail follows the blade; the dragon rears back with its jaws open; moss and leaves fly. The camera shakes slightly."
wan bite      30_f09_bite.png        1280 288 "Close-up: a forest dragon made of bark and moss bites down on a steel sword blade; splinters of bark fly; the dragon's amber eye narrows. The camera is still."
wan retreat   31_c16_retreat.png      480 816 "A wounded forest dragon made of bark and moss backs away across a green meadow, leaving a trail of dark blood; a young man in dark robes watches from the foreground, sword lowered; the grass sways. The camera is still."
wan breath    32_c17_breath.png       640 560 "A young man in dark robes with a cut on his cheek catches his breath and smiles; his shoulders rise and fall; green sparks drift past; his hair stirs in the wind. The camera slowly pushes in."
wan palace    33_c18_palace.png       976 432 "A young man in dark robes stands beside a shining lake before a city of tall crystal spires; the water ripples and sparkles; flowers sway on the shore; light shimmers on the spires. The camera slowly pans."
wan violet    36_c21_eyes.png        1184 320 "Close-up of an elf woman's violet eyes; she blinks slowly and her eyes widen; strands of silver hair drift across her face. The camera is still."
wan border    38_c23_border.png       480 816 "A young man in dark robes and an elf woman with long silver hair walk hand in hand toward a tall wall of pale blue-green light, seen from behind; their hair and robes stir; the light ripples; golden fields lie beyond. The camera follows slowly."
