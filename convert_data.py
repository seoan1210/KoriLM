import os
import re


# =========================
# 설정
# =========================

INPUT_FOLDER = "data"

OUTPUT_RAW = "data.txt"

OUTPUT_CLEAN = "clean_data.txt"



# =========================
# 파일 합치기
# =========================

print("데이터 파일 합치는 중...")


with open(
    OUTPUT_RAW,
    "w",
    encoding="utf-8"
) as out:


    files = sorted(
        os.listdir(INPUT_FOLDER)
    )


    for file in files:


        path = os.path.join(
            INPUT_FOLDER,
            file
        )


        if not os.path.isfile(path):
            continue


        print(
            "추가:",
            file
        )


        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                out.write(
                    f.read()
                )

                out.write(
                    "\n"
                )


        except Exception as e:

            print(
                "오류:",
                file,
                e
            )



print(
    "원본 데이터 생성 완료:",
    OUTPUT_RAW
)



# =========================
# 전처리
# =========================


print(
    "데이터 정리 중..."
)


with open(
    OUTPUT_RAW,
    "r",
    encoding="utf-8"
) as f:

    text=f.read()



text=re.sub(

    r"\{\{.*?\}\}",

    "",

    text,

    flags=re.S

)




text=re.sub(

    r"\[\[|\]\]",

    "",

    text

)



text=re.sub(

    r"<.*?>",

    "",

    text

)


text=re.sub(

    r"\n{3,}",

    "\n\n",

    text

)


text=re.sub(

    r"[<>]",

    "",

    text

)


text=text.strip()



# =========================
# 저장
# =========================


with open(

    OUTPUT_CLEAN,

    "w",

    encoding="utf-8"

) as f:

    f.write(text)



print(
    "완료!"
)

print(
    "최종 데이터:",
    OUTPUT_CLEAN
)

print(
    "크기:",
    round(
        os.path.getsize(OUTPUT_CLEAN)
        /
        1024
        /
        1024,
        2
    ),
    "MB"
)
