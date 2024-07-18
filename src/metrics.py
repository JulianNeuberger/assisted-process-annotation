import dataclasses
import typing

import pet
import base


@dataclasses.dataclass
class Scores:
    p: float
    r: float
    f1: float

    @staticmethod
    def from_stats(stats: "Stats") -> "Scores":
        return Scores(p=stats.precision, r=stats.recall, f1=stats.f1)

    def __add__(self, other):
        if type(other) != Scores:
            raise TypeError(f"Can not add Scores and {type(other)}")
        return Scores(p=self.p + other.p, r=self.r + other.r, f1=self.f1 + other.f1)

    def __truediv__(self, other):
        return Scores(p=self.p / other, r=self.r / other, f1=self.f1 / other)


@dataclasses.dataclass
class Stats:
    num_pred: float
    num_gold: float
    num_ok: float

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        if precision + recall == 0.0:
            return 0
        return 2 * precision * recall / (precision + recall)

    @property
    def precision(self) -> float:
        if self.num_pred == 0 and self.num_gold == 0:
            return 1.0
        elif self.num_pred == 0 and self.num_gold != 0:
            return 0.0
        else:
            return self.num_ok / self.num_pred

    @property
    def recall(self) -> float:
        if self.num_gold == 0 and self.num_pred == 0:
            return 1.0
        elif self.num_gold == 0 and self.num_pred != 0:
            return 0.0
        else:
            return self.num_ok / self.num_gold

    def __add__(self, other):
        if type(other) != Stats:
            raise TypeError(f"Can not add Stats and {type(other)}")
        return Stats(
            num_pred=self.num_pred + other.num_pred,
            num_gold=self.num_gold + other.num_gold,
            num_ok=self.num_ok + other.num_ok,
        )


def relation_f1_stats(
    *,
    predicted_documents: typing.List[pet.PetDocument],
    ground_truth_documents: typing.List[pet.PetDocument],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="relations",
        print_only_tags=print_only_tags,
        verbose=verbose,
    )


def mentions_f1_stats(
    *,
    predicted_documents: typing.List[pet.PetDocument],
    ground_truth_documents: typing.List[pet.PetDocument],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="mentions",
        print_only_tags=print_only_tags,
        verbose=verbose,
    )


def entity_f1_stats(
    *,
    predicted_documents: typing.List[pet.PetDocument],
    ground_truth_documents: typing.List[pet.PetDocument],
    calculate_only_tags: typing.List[str],
    min_num_mentions: int = 1,
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    calculate_only_tags = [t.lower() for t in calculate_only_tags]
    for d in predicted_documents:
        d.entities = [
            e
            for e in d.entities
            if len(e.mention_indices) >= min_num_mentions
            and e.get_tag(d).lower() in calculate_only_tags
        ]

    ground_truth_documents = [d.copy([]) for d in ground_truth_documents]
    for d in ground_truth_documents:
        d.entities = [
            e
            for e in d.entities
            if len(e.mention_indices) >= min_num_mentions
            and e.get_tag(d) in calculate_only_tags
        ]

    return _f1_stats(
        predicted_documents=predicted_documents,
        ground_truth_documents=ground_truth_documents,
        attribute="entities",
        print_only_tags=print_only_tags,
        verbose=verbose,
    )


def _add_to_stats_by_tag(
    stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]],
    get_tag: typing.Callable[[typing.Any], str],
    object_list: typing.Iterable,
    stat: str,
):
    assert stat in ["gold", "pred", "ok"]
    for e in object_list:
        tag = get_tag(e)
        if tag not in stats_by_tag:
            stats_by_tag[tag] = (0, 0, 0)
        prev_stats = stats_by_tag[tag]
        if stat == "gold":
            stats_by_tag[tag] = (prev_stats[0] + 1, prev_stats[1], prev_stats[2])
        elif stat == "pred":
            stats_by_tag[tag] = (prev_stats[0], prev_stats[1] + 1, prev_stats[2])
        else:
            stats_by_tag[tag] = (prev_stats[0], prev_stats[1], prev_stats[2] + 1)
    return stats_by_tag


def _tag(
    document: base.DocumentBase,
    e: typing.Union[
        pet.PetMention,
        pet.PetEntity,
        pet.PetRelation,
    ],
) -> str:
    if isinstance(e, base.HasType):
        return e.type
    if type(e) == pet.PetEntity:
        assert type(document) == pet.PetDocument
        return e.get_tag(document)
    raise AssertionError(f"Unknown type {type(e)}")


def _f1_stats(
    *,
    predicted_documents: typing.List[base.DocumentBase],
    ground_truth_documents: typing.List[base.DocumentBase],
    attribute: str,
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool = False,
) -> typing.Dict[str, Stats]:
    assert attribute in ["mentions", "relations", "entities", "constraints"]
    assert len(predicted_documents) == len(ground_truth_documents)

    stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]] = {}

    for p, t in zip(predicted_documents, ground_truth_documents):
        true_attribute = getattr(t, attribute)
        pred_attribute = getattr(p, attribute)

        true = list(true_attribute)
        pred = list(pred_attribute)
        true_candidates = list(true_attribute)
        ok = []
        non_ok = []
        for cur in pred:
            match: typing.Optional[base.DocumentBase] = None
            if isinstance(cur, base.HasCustomMatch):
                for candidate in true_candidates:
                    if cur.match(candidate):
                        match = candidate
                        break
            else:
                try:
                    match_index = true_candidates.index(cur)
                    match = true_candidates[match_index]
                except ValueError:
                    pass

            if match is not None:
                true_candidates.remove(match)
                ok.append(cur)
                continue
            non_ok.append(cur)
        missing = true_candidates

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(t, e),
            true,
            "gold",
        )
        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(p, e),
            pred,
            "pred",
        )

        _add_to_stats_by_tag(
            stats_by_tag,
            lambda e: _tag(p, e),
            ok,
            "ok",
        )

        if verbose and (len(non_ok) > 0 or len(missing) > 0):
            print_sets(
                t,
                p,
                {
                    "true": true,
                    "pred": pred,
                    # "ok": ok,
                    "non-ok": non_ok,
                    "missing": missing,
                },
                lambda e: _tag(t, e),
                print_only_tags,
            )

    return {
        tag: Stats(num_pred=p, num_gold=g, num_ok=o)
        for tag, (g, p, o) in stats_by_tag.items()
    }


def print_sets(
    true_document: base.DocumentBase,
    pred_document: base.DocumentBase,
    sets: typing.Dict[str, typing.List[base.SupportsPrettyDump]],
    get_tag: typing.Callable[[typing.Any], str],
    print_only_tags: typing.Optional[typing.List[str]],
):
    print(f"=== {true_document.id} " + "=" * 150)
    print(true_document.text)
    print("-" * 100)

    for set_name, values in sets.items():
        document = true_document if set_name in ["true", "missing"] else pred_document
        values = [
            e
            for e in values
            if print_only_tags is None or get_tag(e) in print_only_tags
        ]
        print(f"{len(values)} x {set_name}")
        print("\n".join([e.pretty_dump(document) for e in values]))
        print("-" * 100)
        print()

    print("=" * 150)
    print()
