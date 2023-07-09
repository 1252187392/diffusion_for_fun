eg1 = """
masterpiece,1girl, alternate costume, ass, bangs, very long hair, white hair, white shirt, white skirt, window, bare arms, blue e
yes, blunt bangs, blush, bow, breasts, chair, chromatic aberration, clothes pull, crop top, easy chair, eric (tianqijiang), flower knot, from
below, from side, genshin impact, groin, hair bow, hair ribbon, indoors, kamisato ayaka, long hair, looking at viewer, medium breasts, midriff
, miniskirt, on chair, one knee, panties, parted lips, pink bow, pink panties, pleated skirt, ponytail, red ribbon, revision, ribbon, shirt, s
ide-tie panties, sideboob, sidelighting, signature, skirt, skirt pull, sleeveless, sleeveless shirt, solo, thighs, tress ribbon, underwear
"""
eg2 = """
masterpiece,1girl, animal ears, bangs, breasts, cherry blossoms, cowboy shot, crystalfly (genshin impact), detached sleeves, earrings, falling petals, floating hair, floppy ears, floral print, flower knot, fox ears, genshin impact, hair between eyes, hair ornament, hand up, japanese clothes, jewelry, kusunokinawate, long hair, long sleeves, looking at viewer, medium breasts, nontraditional miko, petals, purple eyes, red skirt, ribbon trim, shirt, sidelocks, skirt, sleeveless, sleeveless shirt, solo, tassel, thighs, turtleneck, white sleeves, wide sleeves, yae miko
"""
eg3 = """
masterpiece,1girl, animal ears, bangs, bare shoulders, blush, breasts, earrings, fantongjun, fox ears, genshin impact, hair ornament, jewelry, large breasts, long hair, looking at viewer, naked towel, open mouth, pink hair, purple eyes, raised eyebrows, sidelocks, solo, thighs, towel, very long hair, yae miko
"""
eg4 = """
modelshoot style,(best quality, masterpiece:1.1), (realistic:1.4),intricate elegant, (highly detailed),sharp focus, dramatic,photorealistic,A beautiful Chinese girl,<lora:nana_v10:1.2:FACEH>,(High Detail), smile,china dress,full body,high-heeled shoes,In the lobby,Screen window, corridor,Curtains of fluttering yarn,((Mottled light and shadow,warm light ,depth of field)),
"""
eg5 = """
(8k, RAW photo, best quality, masterpiece:1.2), (realistic, photo-realistic:1.37),1girl,cute, naked,standing,cityscape, night, rain, wet, professional lighting, photon mapping, radiosity, physically-based rendering
"""

neg_eg1 = 'worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry'

neg_eg2 = '(makeup:1.8),(smooth skin:1.3), anime, illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale:1.2)), futanari, full-package_futanari, penis_from_girl, newhalf, collapsed eyeshadow, multiple eyebrows, vaginas in breasts, pink hair, holes on breasts, fleckles, stretched nipples, gigantic penis, nipples on buttocks, analog, analogphoto, anal sex, signatre, logo, render,'

examples = [[512, 512, eg1.strip(), '', neg_eg1, 137, 1, 1, 25],
            [512, 512,eg2.strip(), '', neg_eg1, 137, 1, 1, 25],
            [512, 512,eg3.strip(), '', neg_eg1, 137, 1, 1, 25],
            [512, 512,eg4.strip(), '', neg_eg1, 137, 1, 1, 25],
            [512, 512,eg5.strip(), '', neg_eg1, 137, 1, 1, 25]
            ]