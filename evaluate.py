import os
import re
import sys
import numpy as np
import mir_eval

# from compute_acc_utils import compute_seg_acc, compute_mark_acc

def get_score(est, ref, key):
    # print(est, ref)
    if key == 'tempo':
        ref = int(ref.split(' ')[0])
        est = int(est.split(' ')[0])
        dt = round(ref * 0.04)
        if abs(ref - est) <= dt or abs(ref - 2*est) <= dt or abs(ref - 1/2 * est) <= dt or abs(ref - 3*est) <= dt or abs(ref - 1/3 * est) <= dt:
            return 1
        else:
            return 0
    if key == 'instrument':
        ref = eval(ref)
        est = eval(est)
        precision = 0
        precision_num = 0
        recall = 0
        recall_num = 0
        for i in est:
            precision_num += 1
            if i in ref:
                precision += 1
        for i in ref:
            recall_num += 1
            if i in est:
                recall += 1
        return recall / recall_num
    if key == 'is_loop':
        return int(est == ref)
    if key == 'key':
        key_dic = {
            'C': 0,
            'C#': 1,
            'D': 2,
            'D#': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'G': 7,
            'G#': 8,
            'A': 9,
            'A#': 10,
            'B': 11,
        }
        mode_est = est[-5:]
        mode_ref = ref[-5:]
        root_est = est[:-5]
        root_ref = ref[:-5]
        if root_est == root_ref and mode_est == mode_ref:
            return 1
        else:
            try:
                if mode_est == mode_ref and abs(key_dic[root_est] - key_dic[root_ref]) == 5:
                    return 0.5
                if root_est == root_est:
                    return 0.2
                if mode_est == 'major' and mode_ref == 'minor':
                    if key_dic[root_est] - key_dic[root_ref] == 3 or key_dic[root_est] - key_dic[root_ref] == -9:
                        return 0.3
                if mode_est == 'minor' and mode_ref == 'major':
                    if key_dic[root_ref] - key_dic[root_est] == 3 or key_dic[root_ref] - key_dic[root_est] == -9:
                        return 0.3
                return 0
            except:
                return 0


def compute_chord_acc(result_path, output_name):
    with open(result_path, "r") as f:
        file_lines = f.readlines()

    groups = "".join(file_lines)
    groups = groups.split("[Question  ]: ")[1:]
    lines = []
    for group in groups:
        question_line = "[Question  ]: " + group.split("[Answer Ref]: ")[0]
        ref_line = (
            "[Answer Ref]: "
            + group.split("[Answer Ref]: ")[1].split("[Answer Est]: ")[0]
        )
        est_line = "[Answer Est]: " + group.split("[Answer Est]: ")[1]
        lines.append(question_line)
        lines.append(ref_line)
        lines.append(est_line)
    print(len(lines), len(lines) / 3)

    refs = [
        re.findall(r"<chord>(.*?)</chord>", line)
        for line in lines
        if len(line.split("[Answer Ref]:")) > 1
    ]
    ests = [
        re.findall(r"<chord>(.*?)</chord>", line)
        for line in lines
        if len(line.split("[Answer Est]:")) > 1
    ]

    error_type1 = 0  # the segment length of ref and est are different
    error_type2 = (
        0  # for the same segment, the number of chord progression is different
    )
    accs = []
    for i, ref_i in enumerate(refs):
        if not len(ref_i) == len(ests[i]):
            error_type1 += 1
            continue
        # compare the two lists
        # the example is ['(0, Fminor), (2.46, D#minor), (5.7, A#minor)']
        # ['(0, Fminor), (2.46, D#minor), (5.7, A#minor)']
        for ref, est in zip(ref_i, ests[i]):
            ref = [c.replace("(", "").replace(")", "") for c in ref.split("), ")]
            est = [c.replace("(", "").replace(")", "") for c in est.split("), ")]
            if not len(ref) == len(est):
                error_type2 += 1
                continue
            ref_times = [float(c.split(", ")[0]) for c in ref]
            est_times = [float(c.split(", ")[0]) for c in est]
            ref_chords = [c.split(", ")[1] for c in ref]
            est_chords = [c.split(", ")[1] for c in est]
            # for times, if the difference is less than 70ms, we consider it as the same chord
            compare = [
                (abs(ref_time - est_time) < 0.5) and (ref_chord == est_chord)
                for ref_time, est_time, ref_chord, est_chord in zip(
                    ref_times, est_times, ref_chords, est_chords
                )
            ]
            # print(ref_times)
            # print(est_times)
            # print(ref_chords)
            # print(est_chords)
            # print(compare)

            for c in compare:
                accs.append(c)

    total_acc = sum(accs) / len(accs)

    print(f"[Total accuracy: {round(total_acc, 2)}]")
    print("[error type 1]: ", error_type1, "[error type 2]: ", error_type2)

    output_path = os.path.join("dataset/new_dataset/evaluation", output_name)
    # with open(output_path, "w") as f:
    #     f.write(f"[Total accuracy {round(total_acc, 2)}]")
    #     f.write("\n")
    #     f.write("[error type 1]: " + str(error_type1))
    #     f.write("[error type 2]: " + str(error_type2))


def compute_key_acc(result_path, output_name):
    with open(result_path, "r") as f:
        file_lines = f.readlines()

    groups = "".join(file_lines)
    groups = groups.split("[Question  ]: ")[1:]
    lines = []
    for group in groups:
        question_line = "[Question  ]: " + group.split("[Answer Ref]: ")[0]
        ref_line = (
            "[Answer Ref]: "
            + group.split("[Answer Ref]: ")[1].split("[Answer Est]: ")[0]
        )
        est_line = "[Answer Est]: " + group.split("[Answer Est]: ")[1]
        lines.append(question_line)
        lines.append(ref_line)
        lines.append(est_line)
    print(len(lines), len(lines) / 3)

    # find all the key signatures within <key> and </key>
    refs = [
        re.findall(r"<key>(.*?)</key>", line)
        for line in lines
        if len(line.split("[Answer Ref]:")) > 1
    ]
    ests = [
        re.findall(r"<key>(.*?)</key>", line)
        for line in lines
        if len(line.split("[Answer Est]:")) > 1
    ]

    # only need one accuracy for each group
    error = 0
    invalid_key = 0
    accs = []
    weigted_scores = []
    sum_scores = 0
    cnt = 0
    for i, ref_i in enumerate(refs):
        if not len(ref_i) == len(ests[i]):
            error += 1
            continue
        # compare the two lists
        # the example is ['C minor', 'C minor'] ['C minor', 'C minor']
        compare = [ref == est for ref, est in zip(ref_i, ests[i])]
        for c in compare:
            accs.append(c)
        for ref, est in zip(ref_i, ests[i]):
            sum_scores += get_score(est, ref, 'key')
            cnt += 1

    print('key acc:', sum_scores / cnt)
            # try:
            #     weigted_score = mir_eval.key.weighted_score(ref, est)
            #     weigted_scores.append(weigted_score)
            # except ValueError:
            #     invalid_key += 1

    # total_acc = sum(accs) / len(accs)
    # total_weigted_score = sum(weigted_scores) / len(weigted_scores)

    # print(
    #     f"[Total accuracy/weigted_score: {round(total_acc, 2)}/{round(total_weigted_score, 2)}]"
    # )
    # print("[error]: ", error, "[invalid_key]: ", invalid_key)

    # output_path = os.path.join("dataset/new_dataset/evaluation", output_name)
    # with open(output_path, "w") as f:
    #     f.write(
    #         f"[Total accuracy/weigted_score {round(total_acc, 2)}/{round(total_weigted_score, 2)}]"
    #     )
    #     f.write("\n")
    #     f.write("[error]: " + str(error))


def compute_instruments_acc(result_path, output_name):
    with open(result_path, "r") as f:
        file_lines = f.readlines()

    groups = "".join(file_lines)
    groups = groups.split("[Question  ]: ")[1:]
    lines = []
    for group in groups:
        question_line = "[Question  ]: " + group.split("[Answer Ref]: ")[0]
        ref_line = (
            "[Answer Ref]: "
            + group.split("[Answer Ref]: ")[1].split("[Answer Est]: ")[0]
        )
        est_line = "[Answer Est]: " + group.split("[Answer Est]: ")[1]
        lines.append(question_line)
        lines.append(ref_line)
        lines.append(est_line)
    print(len(lines), len(lines) / 3)

    refs = [
        re.findall(r"<instruments>(.*?)</instruments>", line)
        for line in lines
        if len(line.split("[Answer Ref]:")) > 1
    ]
    ests = [
        re.findall(r"<instruments>(.*?)</instruments>", line)
        for line in lines
        if len(line.split("[Answer Est]:")) > 1
    ]

    f1s = []
    precisions = []
    recalls = []
    error = 0
    for i in range(len(refs)):
        if not len(refs[i]) == len(ests[i]):
            error += 1
            continue

        for j in range(len(refs[i])):
            refs[i][j] = refs[i][j].split(", ")
            ests[i][j] = ests[i][j].split(", ")
            # compare the two lists
            # the example is ['Erhu', 'ElectricPiano', 'Cello', 'OrganBass', 'Drums'] ['Erhu', 'Cello', 'OrganBass', 'Drums']
            tp = len(set(refs[i][j]) & set(ests[i][j]))
            fp = len(set(ests[i][j]) - set(refs[i][j]))
            fn = len(set(refs[i][j]) - set(ests[i][j]))
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

    total_precision = sum(precisions) / len(precisions)
    total_recall = sum(recalls) / len(recalls)
    total_f1 = sum(f1s) / len(f1s)

    print(
        f"[Total precision/recall/f1: {round(total_precision, 2)}/{round(total_recall, 2)}/{round(total_f1, 2)}]"
    )
    print("[error]: ", error)

    output_path = os.path.join("dataset/new_dataset/evaluation", output_name)
    # with open(output_path, "w") as f:
    #     f.write(
    #         f"[Total precision/recall/f1 {round(total_precision, 2)}/{round(total_recall, 2)}/{round(total_f1, 2)}]"
    #     )
    #     f.write("\n")
    #     f.write("[error]: " + str(error))


def align_bmp(est, ref):
    dt = round(ref * 0.04)
    return (
        np.abs(ref - est) <= dt,
        np.abs(ref - est) <= dt
        or np.abs(ref - est * 2) <= dt
        or np.abs(ref - est / 2) <= dt
        or np.abs(ref - est * 3) <= dt
        or np.abs(ref - est / 3) <= dt,
    )


def compute_tempo_acc(result_path, output_name):
    with open(result_path, "r") as f:
        file_lines = f.readlines()

    groups = "".join(file_lines)
    groups = groups.split("[Question  ]: ")[1:]
    lines = []
    for group in groups:
        question_line = "[Question  ]: " + group.split("[Answer Ref]: ")[0]
        ref_line = (
            "[Answer Ref]: "
            + group.split("[Answer Ref]: ")[1].split("[Answer Est]: ")[0]
        )
        est_line = "[Answer Est]: " + group.split("[Answer Est]: ")[1]
        lines.append(question_line)
        lines.append(ref_line)
        lines.append(est_line)
    print(len(lines), len(lines) / 3)

    refs = [
        re.findall(r"(\d+)\s*bpm", line)
        for line in lines
        if len(line.split("[Answer Ref]:")) > 1
    ]
    ests = [
        re.findall(r"(\d+)\s*bpm", line)
        for line in lines
        if len(line.split("[Answer Est]:")) > 1
    ]

    results = {}
    error = 0
    for i in range(len(refs)):
        if not len(refs[i]) == len(ests[i]):
            error += 1
            continue
        c = str(len(refs[i]))
        if c not in results:
            results[c] = np.zeros([len(refs[i]), 3])

        for j in range(len(refs[i])):
            results[c][j, 0] += 1
            refs[i][j] = round(float(refs[i][j]))
            ests[i][j] = round(float(ests[i][j]))
            acc1, acc2 = align_bmp(ests[i][j], refs[i][j])
            if acc1:
                results[c][j, 1] += 1
            if acc2:
                results[c][j, 2] += 1

    eval = []
    acc_1 = 0.0
    acc_2 = 0.0
    acc_n = 0.0
    for c in results:
        # print(f"[A group of {c} samples]")
        eval.append(f"[A group of {c} samples]")
        for i in range(int(c)):
            res = f"({int(results[c][i, 1])}, {int(results[c][i, 2])})/{int(results[c][i, 0])}, acc1: {round(results[c][i, 1] / results[c][i, 0], 2)}; acc2: {round(results[c][i, 2] / results[c][i, 0], 2)}"
            eval.append(res)
            # print(res)
        acc_n += int(results[c][0, 0])
        acc_1 += int(results[c][0, 1])
        acc_2 += int(results[c][0, 2])

    eval.append(
        f"[Total Tempo Accuracy {int(acc_1), int(acc_2)} / {int(acc_n)}]: [acc1: {round(acc_1 / acc_n, 2)}] [acc2: {round(acc_2 / acc_n, 2)}]"
    )
    print(eval[-1])

    # output_path = os.path.join("dataset/new_dataset/evaluation", output_name)
    # with open(output_path, "w") as f:
    #     f.write("\n".join(eval))
    # print("[error]: ", error)


# def compute_contrast_tempo_acc(result_path, output_name):
#     with open(result_path, "r") as f:
#         lines = f.readlines()
#     lines = [line.rstrip() for line in lines]

#     refs = [
#         line.split("<faster>")[-1].split("</faster")[0]
#         for line in lines
#         if len(line.split("[Answer Ref]:")) > 1
#     ]
#     ests = [
#         line.split("<faster>")[-1].split("</faster")[0]
#         for line in lines
#         if len(line.split("[Answer Est]:")) > 1
#     ]
#     acc = [refs[i] == ests[i] for i in range(len(refs))]
#     print(
#         f"[Total Accuracy {int(sum(acc))} / {len(acc)}]: {round(sum(acc) / len(acc), 2)}"
#     )


if __name__ == "__main__":
    # suffix = "_1008(4300 aam)"  # "_bs-4-16_ckpt650"
    # result_path = f"/datapool/data2/home/ruihan/storage/debug/all_m4m/revising/m4m_dataset/dataset/results/QA_test{suffix}"
    suffix = ''
    result_path = '/datapool/data2/home/ruihan/storage/debug/all_m4m/revising/m4m_dataset/dataset/results/QA_change_key'
    result_path = '/datapool/data2/home/ruihan/storage/debug/all_m4m/inference_results'
    result_path = '/datapool/data2/home/ruihan/storage/debug/all_m4m/revising/m4m_dataset/dataset/results/QA_test_1007(4300)'

    print(f"Computing accuracy for {result_path}")

    print("=====================================================")
    print("Computing accuracy for tempo:")
    compute_tempo_acc(result_path, f"tempo_acc{suffix}.txt")

    print("=====================================================")
    print("Computing accuracy for key:")
    compute_key_acc(result_path, f"key_acc{suffix}.txt")

    print("=====================================================")
    print("Computing accuracy for instruments:")
    compute_instruments_acc(result_path, f"instruments_acc{suffix}.txt")

    print("=====================================================")
    print("Computing accuracy for chords:")
    compute_chord_acc(result_path, f"chord_acc{suffix}.txt")

    # print("=====================================================")
    # compute_seg_acc(result_path, f"seg_acc{suffix}.txt")
    # print("=====================================================")
    # compute_mark_acc(result_path, f"mark_acc{suffix}.txt")