import collections
import dataclasses
import json
import typing

import base


@dataclasses.dataclass
class PetDocument(
    base.DocumentBase, base.HasMentions["PetMention"], base.HasRelations["PetRelation"]
):
    category: str
    name: str
    tokens: typing.List["PetToken"]
    entities: typing.List["PetEntity"]

    @property
    def sentences(self) -> typing.List[typing.List["PetToken"]]:
        ret = []
        last_id = None
        for token in self.tokens:
            if token.sentence_index != last_id:
                last_id = token.sentence_index
                ret.append([])
            ret[-1].append(token)
        return ret

    def copy(self, clear: typing.List[str]) -> "PetDocument":
        return PetDocument(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=[t.copy() for t in self.tokens],
            mentions=[] if "mentions" in clear else [m.copy() for m in self.mentions],
            relations=(
                [] if "relations" in clear else [r.copy() for r in self.relations]
            ),
            entities=[] if "entities" in clear else [e.copy() for e in self.entities],
        )

    def __add__(self, other: "PetDocument"):
        assert self.id == other.id
        assert self.tokens == other.tokens

        new_mentions = self.mentions
        new_mention_ids = {}
        for i, mention in enumerate(other.mentions):
            if mention not in new_mentions:
                new_mention_ids[i] = len(new_mentions)
                new_mentions.append(mention)
            else:
                new_mention_ids[i] = new_mentions.index(mention)

        new_entities = self.entities
        for entity in other.entities:
            new_entity = PetEntity(
                mention_indices=tuple(
                    new_mention_ids[i] for i in entity.mention_indices
                ),
                document=self,
            )
            if new_entity not in new_entities:
                new_entities.append(new_entity)

        new_relations = self.relations
        for relation in other.relations:
            new_relation = PetRelation(
                type=relation.type,
                head_mention_index=new_mention_ids[relation.head_mention_index],
                tail_mention_index=new_mention_ids[relation.tail_mention_index],
                document=self,
            )
            if new_relation not in new_relations:
                new_relations.append(new_relation)

        return PetDocument(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=self.tokens,
            mentions=new_mentions,
            entities=new_entities,
            relations=new_relations,
        )


@dataclasses.dataclass(frozen=True)
class PetMention(base.HasType, base.SupportsPrettyDump[PetDocument]):
    token_document_indices: typing.Tuple[int, ...]

    def copy(self) -> "PetMention":
        return PetMention(
            type=self.type.strip().lower(),
            token_document_indices=tuple(i for i in self.token_document_indices),
        )

    def text(self, document: "PetDocument") -> str:
        return " ".join([document.tokens[i].text for i in self.token_document_indices])

    def pretty_dump(self, document: "PetDocument") -> str:
        return f"{self.type}, '{self.text(document)}', {self.token_document_indices}"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PetMention):
            return False
        if self.type.lower() != o.type.lower():
            return False
        return sorted(self.token_document_indices) == sorted(o.token_document_indices)

    def __hash__(self) -> int:
        element_counts = collections.Counter(self.token_document_indices)
        cur = hash(frozenset(element_counts.items()))
        cur += hash(self.type.lower())
        return cur

    def relaxed_match(self, o: object):
        if not isinstance(o, PetMention):
            return False
        if self.type.lower() != o.type.lower():
            return False
        if any([i in o.token_document_indices for i in self.token_document_indices]):
            return True
        return False

    def exact_match(self, o: object):
        if not isinstance(o, PetMention):
            return False
        if self.type.lower() != o.type.lower():
            return False
        for i, j in zip(self.token_document_indices, o.token_document_indices):
            if i != j:
                return False
        return True


@dataclasses.dataclass(frozen=True)
class PetEntity(base.SupportsPrettyDump[PetDocument], base.HasCustomMatch):
    mention_indices: typing.Tuple[int, ...]

    document: PetDocument

    def copy(self) -> "PetEntity":
        return PetEntity(
            mention_indices=tuple(i for i in self.mention_indices),
            document=self.document,
        )

    def get_tag(self, document: "PetDocument") -> str:
        if max(self.mention_indices) >= len(document.mentions):

            print(json.dumps(PetDictExporter().export_document(document)))
            raise AssertionError(
                f"Entity mentions out of bounds: {self.mention_indices}, {document.mentions}"
            )
        tags = set(document.mentions[i].type for i in self.mention_indices)
        if len(tags) > 1:
            print(f"Entity has mentions of mixed ner tags: {tags}")
        return list(tags)[0]

    def pretty_dump(self, document: PetDocument) -> str:
        formatted_mentions = [
            f"{i}: '{m.text(document)}' ({m.token_document_indices})"
            for i, m in [(i, document.mentions[i]) for i in self.mention_indices]
        ]
        return ", ".join(formatted_mentions)

    def match(self, o: object) -> bool:
        if not isinstance(o, PetEntity):
            return False
        if len(self.mention_indices) != len(o.mention_indices):
            return False

        self_candidates = [self.document.mentions[i] for i in self.mention_indices]
        other_candidates = [o.document.mentions[i] for i in o.mention_indices]
        for self_candidate in self_candidates:
            matched = False
            for other_candidate in other_candidates:
                if self_candidate.relaxed_match(other_candidate):
                    matched = True
                    other_candidates.remove(other_candidate)
                    break
            if not matched:
                return False
        return len(other_candidates) == 0

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PetEntity):
            return False
        if len(self.mention_indices) != len(o.mention_indices):
            return False
        return sorted(self.mention_indices) == sorted(o.mention_indices)

    def __hash__(self):
        element_counts = collections.Counter(self.mention_indices)
        return hash(frozenset(element_counts.items()))


@dataclasses.dataclass(frozen=True, eq=True)
class PetRelation(
    base.HasType, base.SupportsPrettyDump[PetDocument], base.HasCustomMatch
):
    head_mention_index: int
    tail_mention_index: int

    document: PetDocument

    def copy(self) -> "PetRelation":
        return PetRelation(
            head_mention_index=self.head_mention_index,
            tail_mention_index=self.tail_mention_index,
            type=self.type.lower().strip(),
            document=self.document,
        )

    def pretty_dump(self, document: PetDocument) -> str:
        head = document.mentions[self.head_mention_index].pretty_dump(document)
        tail = document.mentions[self.tail_mention_index].pretty_dump(document)
        return f"{head} -{self.type}-> {tail}"

    def match(self, o: object) -> bool:
        if not isinstance(o, PetRelation):
            return False
        if self.type.lower() != o.type.lower():
            return False

        self_head = self.document.mentions[self.head_mention_index]
        other_head = o.document.mentions[o.head_mention_index]
        if not self_head.relaxed_match(other_head):
            return False

        self_tail = self.document.mentions[self.tail_mention_index]
        other_tail = o.document.mentions[o.tail_mention_index]
        if not self_tail.relaxed_match(other_tail):
            return False

        return True


@dataclasses.dataclass
class PetToken:
    text: str
    index_in_document: int
    pos_tag: str
    sentence_index: int

    def char_indices(self, document: PetDocument) -> typing.Tuple[int, int]:
        start = 0
        for i, other in enumerate(document.tokens):
            if other == self:
                return start, start + len(self.text)
            start += len(other.text) + 1
        raise AssertionError("Token text not found in document")

    def copy(self) -> "PetToken":
        return PetToken(
            text=self.text,
            index_in_document=self.index_in_document,
            pos_tag=self.pos_tag,
            sentence_index=self.sentence_index,
        )


class PetJsonExporter:
    def __init__(self, path: str):
        self._dict_exporter = PetDictExporter()
        self._path = path

    def export(self, documents: typing.List[PetDocument]):
        json_lines = []
        for document in documents:
            document_as_json = json.dumps(self._dict_exporter.export_document(document))
            json_lines.append(document_as_json)
        with open(self._path, "w", encoding="utf8") as f:
            f.write("\n".join(json_lines))


class PetDictExporter:
    def export_document(self, document: PetDocument) -> typing.Dict:
        return {
            "text": document.text,
            "name": document.name,
            "id": document.id,
            "category": document.category,
            "tokens": list(map(self.export_token, document.tokens)),
            "mentions": list(map(self.export_mention, document.mentions)),
            "entities": list(map(self.export_entity, document.entities)),
            "relations": list(map(self.export_relation, document.relations)),
        }

    def export_token(self, token: PetToken) -> typing.Dict:
        return {
            "text": token.text,
            "indexInDocument": token.index_in_document,
            "posTag": token.pos_tag,
            "sentenceIndex": token.sentence_index,
        }

    def export_mention(self, mention: PetMention) -> typing.Dict:
        return {
            "type": mention.type,
            "tokenDocumentIndices": list(mention.token_document_indices),
        }

    def export_relation(self, relation: PetRelation) -> typing.Dict:
        return {
            "headMentionIndex": relation.head_mention_index,
            "tailMentionIndex": relation.tail_mention_index,
            "type": relation.type,
        }

    def export_entity(self, entity: PetEntity) -> typing.Dict:
        return {"mentionIndices": entity.mention_indices}


class NewPetFormatImporter(base.BaseImporter[PetDocument]):
    class DictImporter:
        @staticmethod
        def read_tokens_from_dict(
            json_tokens: typing.List[typing.Dict],
        ) -> typing.List[PetToken]:
            tokens = []
            for i, json_token in enumerate(json_tokens):
                tokens.append(
                    PetToken(
                        text=json_token["text"],
                        pos_tag=json_token["posTag"],
                        index_in_document=i,
                        sentence_index=json_token["sentenceIndex"],
                    )
                )
            return tokens

        @staticmethod
        def read_mentions_from_dict(
            json_mentions: typing.List[typing.Dict],
        ) -> typing.List[PetMention]:
            mentions = []
            for json_mention in json_mentions:
                mention = NewPetFormatImporter.DictImporter.read_mention_from_dict(
                    json_mention
                )
                mentions.append(mention)
            return mentions

        @staticmethod
        def read_entities_from_dict(
            json_entities: typing.List[typing.Dict], document: PetDocument
        ) -> typing.List[PetEntity]:
            entities = []
            for json_entity in json_entities:
                entity = NewPetFormatImporter.DictImporter.read_entity_from_dict(
                    json_entity, document
                )
                entities.append(entity)
            return entities

        @staticmethod
        def read_mention_from_dict(json_mention: typing.Dict) -> PetMention:
            return PetMention(
                type=json_mention["type"].lower().strip(),
                token_document_indices=tuple(json_mention["tokenDocumentIndices"]),
            )

        @staticmethod
        def read_entity_from_dict(
            json_entity: typing.Dict, document: PetDocument
        ) -> PetEntity:
            return PetEntity(json_entity["mentionIndices"], document)

        @staticmethod
        def read_relations_from_dict(
            json_relations: typing.List[typing.Dict], document: PetDocument
        ) -> typing.List[PetRelation]:
            relations = []
            for json_relation in json_relations:
                relations.append(
                    NewPetFormatImporter.DictImporter.read_relation_from_dict(
                        json_relation, document
                    )
                )
            return relations

        @staticmethod
        def read_relation_from_dict(
            relation_dict: typing.Dict, document: PetDocument
        ) -> PetRelation:
            head_mention_index = relation_dict["headMentionIndex"]
            tail_mention_index = relation_dict["tailMentionIndex"]
            return PetRelation(
                head_mention_index=head_mention_index,
                tail_mention_index=tail_mention_index,
                type=relation_dict["type"].lower().strip(),
                document=document,
            )

    def __init__(self, file_path: str):
        self._file_path = file_path

    def do_import(self) -> typing.List[PetDocument]:
        documents: typing.List[PetDocument] = []
        with open(self._file_path, "r", encoding="utf8") as f:
            for json_line in f:
                json_data = json.loads(json_line)
                documents.append(self.read_document_from_json(json_data))
        return documents

    @staticmethod
    def read_document_from_file(file_path: str) -> PetDocument:
        with open(file_path, "r", encoding="utf8") as f:
            return NewPetFormatImporter.read_document_from_json(json.load(f))

    @staticmethod
    def read_document_from_json(json_data: typing.Dict) -> PetDocument:
        document = PetDocument(
            name=json_data["name"],
            text=json_data["text"],
            id=json_data["id"],
            category=json_data["category"],
            tokens=[],
            mentions=[],
            relations=[],
            entities=[],
        )

        mentions = NewPetFormatImporter.DictImporter.read_mentions_from_dict(
            json_data["mentions"]
        )
        entities = NewPetFormatImporter.DictImporter.read_entities_from_dict(
            json_data["entities"], document
        )
        relations = NewPetFormatImporter.DictImporter.read_relations_from_dict(
            json_data["relations"], document
        )
        tokens = NewPetFormatImporter.DictImporter.read_tokens_from_dict(
            json_data["tokens"]
        )

        document.tokens = tokens
        document.mentions = mentions
        document.entities = entities
        document.relations = relations

        return document

